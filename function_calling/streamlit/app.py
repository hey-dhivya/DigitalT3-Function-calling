import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from typing import Callable, Generator, Optional
import re

import streamlit as st
import yaml
from youtube_transcript_api import YouTubeTranscriptApi as yta

# Directory and environment setup
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from function_calling.src.function_calling import FunctionCallingLlm  # type: ignore
from function_calling.src.tools import calculator, get_time, python_repl, query_db, translate  # type: ignore
from utils.visual.env_utils import initialize_env_variables

logging.basicConfig(level=logging.INFO)

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')

# Tool mapping of available tools
TOOLS = {
    'get_time': get_time,
    'calculator': calculator,
    'python_repl': python_repl,
    'query_db': query_db,
    'translate': translate,
}


def load_config():
    with open(CONFIG_PATH, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


config = load_config()
prod_mode = config.get('prod_mode', False)
additional_env_vars = config.get('additional_env_vars', None)


@contextmanager
def st_capture(output_func: Callable[[str], None]) -> Generator:
    """
    Context manager to catch stdout and send it to an output Streamlit element.
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string: str) -> int:
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write  # type: ignore
        yield


def set_fc_llm(tools: list) -> None:
    """
    Set the FunctionCallingLlm object with the selected tools.
    """
    set_tools = [TOOLS[name] for name in tools]
    st.session_state.fc = FunctionCallingLlm(set_tools)


def extract_transcript(youtube_link: str) -> str:
    """
    Extract transcript from the given YouTube link.
    """
    video_id = None
    video_id_patterns = [
        r"v=([a-zA-Z0-9_-]+)",  # Standard URL format
        r"youtu\.be/([a-zA-Z0-9_-]+)",  # Short URL format
    ]

    for pattern in video_id_patterns:
        match = re.search(pattern, youtube_link)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        return "Invalid YouTube link. Please ensure it is a valid URL with a video ID."

    try:
        data = yta.get_transcript(video_id)
        transcript = " ".join([entry['text'] for entry in data if 'text' in entry])
        return transcript.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def handle_common_questions(question: str) -> str:
    """
    Handle common questions and greetings.
    """
    common_responses = {
        'hi': "Hello! How can I assist you today?",
        'hello': "Hi there! How can I help you?",
        'can you help me': "Of course! What do you need help with?",
    }

    question_lower = question.lower()
    for key in common_responses:
        if key in question_lower:
            return common_responses[key]

    return None


def segment_transcript_by_keywords(transcript: str, keywords: list) -> dict:
    """
    Segment the transcript by provided keywords.
    """
    sections = {}
    for i, keyword in enumerate(keywords):
        # Create a regex pattern for each keyword to capture relevant segments
        pattern = re.compile(rf'({keyword}.*?)(?=(?:{"|".join(keywords[i + 1:])}|$))', re.DOTALL)
        match = pattern.search(transcript)
        if match:
            sections[keyword] = match.group(1).strip()
            logging.debug(f"Segment for '{keyword}': {sections[keyword]}")
        else:
            sections[keyword] = ""

    return sections


def answer_from_transcript(question: str, transcript: str) -> str:
    """
    Answer the user's question based on keywords in the transcript,
    or handle general Python questions directly.
    """
    # Handle common questions first
    common_response = handle_common_questions(question)
    if common_response:
        return common_response

    # List of keywords related to coding topics
    keywords = ['code', 'functions', 'installation', 'examples', 'syntax', 'logic']

    # Segment the transcript using the keywords
    segments = segment_transcript_by_keywords(transcript, keywords)

    # Check for relevant keyword in the question
    question_lower = question.lower()
    for keyword in keywords:
        if keyword in question_lower:
            if segments.get(keyword):
                return f"Here is the information I found about {keyword}:\n\n{segments[keyword]}"

    # Handle general Python knowledge questions
    if "list" in question_lower and "data types" in question_lower:
        return "A list in Python can hold items of any data type, including:\n\n- Integers\n- Floats\n- Strings\n- Lists (nested)\n- Tuples\n- Dictionaries\n- Booleans\n- Objects (instances of classes)\n- NoneType"

    # If no relevant information is found in the transcript or general knowledge
    return "I couldn't find any information related to your question in the transcript, but feel free to ask about Python or coding topics!"

def explain_code_snippet(code_snippet: str) -> str:
    """
    Provide an explanation for a given code snippet.
    """
    # Replace this with a real function or API that can explain code snippets
    return f"Explanation of the code: {code_snippet}"


def handle_userinput(user_question: Optional[str]) -> None:
    """
    Handle user input and generate a response, also update chat UI in Streamlit app.
    """
    if user_question:
        with st.spinner('Processing...'):
            response = answer_from_transcript(user_question,
                                              st.session_state.transcript) if st.session_state.transcript else "No transcript available. Please provide a YouTube link to extract the transcript."

        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)

    # Display chat history
    for ques, ans in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message('ai', avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png'):
            st.write(f'{ans}')


def main() -> None:
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    initialize_env_variables(prod_mode, additional_env_vars)

    # Initialize session state if not already done
    if 'fc' not in st.session_state:
        st.session_state.fc = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'tools' not in st.session_state:
        st.session_state.tools = ['get_time', 'python_repl', 'query_db']
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 5
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = False  # Ensure input is enabled by default
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""

    st.title(':orange[SambaNova] AI VIDQUEST')

    # Sidebar for YouTube video link input
    st.sidebar.title("Upload YouTube Video Link")
    youtube_link = st.sidebar.text_input("YouTube video link", placeholder="Paste YouTube link here")

    if youtube_link:
        # Extract and display the transcript
        transcript = extract_transcript(youtube_link)
        st.session_state.transcript = transcript
        st.sidebar.subheader("Transcript")
        st.sidebar.write(transcript)
        st.session_state.input_disabled = False  # Enable input after transcript is loaded

    with st.expander('Additional settings', expanded=False):
        st.markdown('**Interaction options**')

        st.markdown('**Reset chat**')
        st.markdown('**Note:** Resetting the chat will clear all interactions history')
        if st.button('Reset conversation'):
            st.session_state.chat_history = []
            st.session_state.input_disabled = True  # Disable input until a new transcript is loaded

    user_question = st.chat_input('Ask something', disabled=st.session_state.input_disabled, key='TheChatInput')
    handle_userinput(user_question)


if __name__ == '__main__':
    main()