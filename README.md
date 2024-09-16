🎥 AI VIDQUEST - Intelligent Q&A Generation from YouTube Videos


Welcome to the AI VIDQUEST application powered by the SambaNova AI Starter Kit! This app allows users to upload a YouTube video link, extract its transcript, and interact with a context-aware chatbot to ask questions related to the transcript or general Python coding topics. It also integrates several tools for function calling, enabling dynamic answers to user queries.

![Screenshot 2024-09-15 195132](https://github.com/user-attachments/assets/3e1cafea-aa4d-4e1b-8d50-1e80844fb253)

## 🛠 Features

- **YouTube Transcript Extraction**: Upload a YouTube video link and extract the full transcript for analysis.

- **Context-Aware Chatbot**: Ask questions based on the extracted transcript or general Python topics.

- **Tool Integration**: Function calling tools like calculator, get_time, and python_repl.

- **Transcript Segmentation**: Automatically segments the transcript by coding-related keywords to provide more relevant answers.
## 🧑‍💻 Key Files

- **app.py**: The main Streamlit app file.

- **function_calling/src/**: The directory containing function calling logic and tools.

- **config.yaml**: Configuration file for setting production mode and additional environment variables.

## 📚 Usage

- **Upload YouTube Link**: Paste a YouTube video link into the input box in the sidebar. The app will automatically extract the transcript and display it below the link.

- **Ask Questions**: Once the transcript is available, you can ask questions in the chat input box related to the video content or general Python programming topics.
