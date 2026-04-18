from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import textwrap
import time

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="HK Policy AI Assistant", page_icon="✨")

# -----------------------------------------------------------------------------
# Set things up.

executor = ThreadPoolExecutor(max_workers=5)

MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
# Number of context chunks to retrieve per RAG source (set > 0 to enable when
# the retrieval backends are wired up).
DOCS_CONTEXT_LEN = 10
EXTRA_CONTEXT_LEN = 10
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=3)

PROJECT_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = PROJECT_DIR / "hk_policy_chunks.json"
INDEX_DIR = PROJECT_DIR / ".rag_index"
INDEX_FILE = INDEX_DIR / "hk_policy.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
EMBEDDING_BATCH_SIZE = 100

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS = textwrap.dedent("""
    - You are a helpful AI assistant that answers questions about Hong Kong
      government policies, regulations, and public affairs.
    - You will be given extra information provided inside tags like this
      <foo></foo>.
    - Use context and history to provide a coherent answer.
    - Use markdown such as headers (starting with ##), code blocks, bullet
      points, indentation for sub bullets, and backticks for inline code.
    - Don't start the response with a markdown header.
    - Be clear and accurate. Cite sources when available in the context.
    - Don't say things like "according to the provided context".
    - If you are unsure, say so rather than guessing.
""")

SUGGESTIONS = {
    ":blue[:material/local_library:] 什麼是《基本法》?": (
        "簡介香港《基本法》的主要內容和重要性。"
    ),
    ":green[:material/gavel:] 香港的行政架構": (
        "香港特別行政區的行政架構是怎樣的？行政長官的職責是什麼？"
    ),
    ":orange[:material/apartment:] 房屋政策": (
        "香港目前的房屋政策有哪些？公營房屋申請資格是什麼？"
    ),
    ":violet[:material/balance:] 法律體系": (
        "香港採用什麼法律體系？與內地法律有何不同？"
    ),
    ":red[:material/public:] 社會福利政策": (
        "香港有哪些主要的社會福利政策和援助計劃？"
    ),
}


def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags.

    For example, this:

        build_prompt(foo="1\n2\n3", bar="4\n5\n6")

    ...returns:

        '''
        <foo>
        1
        2
        3
        </foo>
        <bar>
        4
        5
        6
        </bar>
        '''
    """
    prompt = []

    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")

    prompt_str = "\n".join(prompt)

    return prompt_str


# Just some little objects to make tasks more readable.
TaskInfo = namedtuple("TaskInfo", ["name", "function", "args"])
TaskResult = namedtuple("TaskResult", ["name", "result"])


def build_question_prompt(question):
    """Fetches info from different sources and creates the prompt string."""
    old_history = st.session_state.messages[:-HISTORY_LENGTH]
    recent_history = st.session_state.messages[-HISTORY_LENGTH:]

    if recent_history:
        recent_history_str = history_to_text(recent_history)
    else:
        recent_history_str = None

    # Fetch information from different RAG sources in parallel.
    task_infos = []

    if SUMMARIZE_OLD_HISTORY and old_history:
        task_infos.append(
            TaskInfo(
                name="old_message_summary",
                function=generate_chat_summary,
                args=(old_history,),
            )
        )

    if DOCS_CONTEXT_LEN:
        task_infos.append(
            TaskInfo(
                name="policy_documents",
                function=search_relevant_docs,
                args=(question,),
            )
        )

    if EXTRA_CONTEXT_LEN:
        task_infos.append(
            TaskInfo(
                name="extra_context",
                function=search_extra_context,
                args=(question,),
            )
        )

    results = executor.map(
        lambda task_info: TaskResult(
            name=task_info.name,
            result=task_info.function(*task_info.args),
        ),
        task_infos,
    )

    context = {name: result for name, result in results}
    # Drop empty context entries so they don't clutter the prompt.
    context = {k: v for k, v in context.items() if v}

    prompt = build_prompt(
        instructions=INSTRUCTIONS,
        **context,
        recent_messages=recent_history_str,
        question=question,
    )
    return prompt, context


def generate_chat_summary(messages):
    """Summarizes the chat history in `messages`."""
    prompt = build_prompt(
        instructions="Summarize this conversation as concisely as possible.",
        conversation=history_to_text(messages),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)


def _embed_texts(texts):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _load_chunks_data():
    if not CHUNKS_FILE.exists():
        return []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_and_persist_index(chunks_data):
    if not chunks_data:
        return None, []

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    texts = [item.get("page_content", "") for item in chunks_data]
    metadata = [item.get("metadata", {}) for item in chunks_data]

    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = _embed_texts(batch)
        all_embeddings.extend(batch_embeddings)

    vectors = np.array(all_embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return index, metadata


@st.cache_resource
def get_rag_index():
    if not os.getenv("OPENAI_API_KEY"):
        return None, []

    if INDEX_FILE.exists() and METADATA_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        with METADATA_FILE.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    chunks_data = _load_chunks_data()
    return _build_and_persist_index(chunks_data)


@st.cache_data
def get_chunks_data():
    return _load_chunks_data()


def _extract_years_from_query(query):
    tokens = query.replace("/", " ").replace("-", " ").split()
    years = []
    for token in tokens:
        if token.isdigit() and len(token) == 4 and token.startswith(("19", "20")):
            years.append(token)
    return set(years)


def search_relevant_docs(query):
    """Retrieves relevant chunks with FAISS and returns prompt-ready context."""
    index, metadata = get_rag_index()
    chunks_data = get_chunks_data()

    if index is None or not chunks_data:
        return ""

    query_embedding = _embed_texts([query])[0]
    query_vector = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    k = min(DOCS_CONTEXT_LEN, index.ntotal)
    distances, indices = index.search(query_vector, k)

    context_lines = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue

        chunk = chunks_data[idx]
        md = metadata[idx] if idx < len(metadata) else {}
        source_type = md.get("source_type", "Unknown")
        year = md.get("year", "Unknown")
        filename = md.get("filename", "Unknown")
        page = md.get("page", "Unknown")
        score = float(distances[0][rank])
        text = chunk.get("page_content", "").strip()

        context_lines.append(
            "\\n".join(
                [
                    f"[Chunk {rank + 1}]",
                    f"source_type={source_type}; year={year}; file={filename}; page={page}; score={score:.4f}",
                    text,
                ]
            )
        )

    return "\n\n".join(context_lines)


def search_extra_context(query):
    """Applies lightweight metadata filtering for targeted policy queries."""
    chunks_data = get_chunks_data()
    if not chunks_data:
        return ""

    query_lower = query.lower()
    years = _extract_years_from_query(query)

    wanted_source = None
    if "budget" in query_lower or "預算" in query or "预算" in query:
        wanted_source = "Budget"
    elif "policy address" in query_lower or "施政報告" in query:
        wanted_source = "Policy Address"

    matched = []
    for item in chunks_data:
        md = item.get("metadata", {})
        source_type = md.get("source_type", "")
        year = str(md.get("year", ""))

        if wanted_source and source_type != wanted_source:
            continue
        if years and year not in years:
            continue

        matched.append(item)
        if len(matched) >= EXTRA_CONTEXT_LEN:
            break

    if not matched:
        return ""

    lines = []
    for i, item in enumerate(matched, start=1):
        md = item.get("metadata", {})
        lines.append(
            "\\n".join(
                [
                    f"[Filtered {i}]",
                    f"source_type={md.get('source_type', 'Unknown')}; year={md.get('year', 'Unknown')}; file={md.get('filename', 'Unknown')}; page={md.get('page', 'Unknown')}",
                    item.get("page_content", "").strip(),
                ]
            )
        )

    return "\n\n".join(lines)


def get_response(prompt):
    """Calls the LLM and returns a streaming text generator."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    def _gen():
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return _gen()


def send_telemetry(**kwargs):
    """Records some telemetry about questions being asked."""
    # TODO: Implement this.
    pass


def show_feedback_controls(message_index):
    """Shows the "How did I do?" control."""
    st.write("")

    with st.popover("How did I do?"):
        with st.form(key=f"feedback-{message_index}", border=False):
            with st.container(gap=None):
                st.markdown(":small[Rating]")
                rating = st.feedback(options="stars")

            details = st.text_area("More information (optional)")

            if st.checkbox("Include chat history with my feedback", True):
                relevant_history = st.session_state.messages[:message_index]
            else:
                relevant_history = []

            ""  # Add some space

            if st.form_submit_button("Send feedback"):
                # TODO: Submit feedback here!
                pass


@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("""
            This AI chatbot is powered by OpenAI and local policy documents.
            Answers may be inaccurate, inefficient, or biased.
            Any use or decisions based on such answers should include reasonable
            practices including human oversight to ensure they are safe,
            accurate, and suitable for your intended purpose. The project is not
            liable for any actions, losses, or damages resulting from the use
            of the chatbot. Do not enter any private, sensitive, personal, or
            regulated data.
        """)


# -----------------------------------------------------------------------------
# Draw the UI.


st.html(div(style=styles(font_size=rem(5), line_height=1))["❉"])

title_row = st.container(
    horizontal=True,
    vertical_alignment="bottom",
)

with title_row:
    st.title(
        "HK Policy AI Assistant",
        anchor=False,
        width="stretch",
    )

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)

has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Show a different UI when the user hasn't asked a question yet.
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Show chat input at the bottom when a question has been asked.
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None

    st.button(
        "Restart",
        icon=":material/refresh:",
        on_click=clear_conversation,
    )

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# Display chat messages from history as speech bubbles.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_feedback_controls(i)

if user_message:
    # When the user posts a message...

    # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
    # display math). The line below fixes it.
    user_message = user_message.replace("$", r"\$")

    # Display message as a speech bubble.
    with st.chat_message("user"):
        st.text(user_message)

    # Display assistant response as a speech bubble.
    with st.chat_message("assistant"):
        with st.spinner("Waiting..."):
            # Rate-limit the input if needed.
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp

            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

            user_message = user_message.replace("'", "")

        # Build a detailed prompt.
        if DEBUG_MODE:
            with st.status("Computing prompt...") as status:
                full_prompt, retrieved_context = build_question_prompt(user_message)
                st.code(full_prompt)
                status.update(label="Prompt computed")
        else:
            with st.spinner("Researching..."):
                full_prompt, retrieved_context = build_question_prompt(user_message)

        # Send prompt to LLM.
        with st.spinner("Thinking..."):
            response_gen = get_response(full_prompt)

        # Put everything after the spinners in a container to fix the
        # ghost message bug.
        with st.container():
            # Stream the LLM response.
            response = st.write_stream(response_gen)

            retrieved_docs = retrieved_context.get("policy_documents", "")
            if retrieved_docs:
                with st.expander("Referenced knowledge chunks", expanded=False):
                    st.text(retrieved_docs)

            # Add messages to chat history.
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Other stuff.
            show_feedback_controls(len(st.session_state.messages) - 1)
            send_telemetry(question=user_message, response=response)
