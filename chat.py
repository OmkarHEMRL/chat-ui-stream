import streamlit as st
import ollama
from openai import OpenAI
from utilities.icon import page_icon
from PyPDF2 import PdfReader  # Updated import

st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def extract_model_names(models_info: list) -> tuple:
    """
    Extracts the model names from the models information.

    :param models_info: A dictionary containing the models' information.
    :return: A tuple containing the model names.
    """
    return tuple(model["name"] for model in models_info["models"])

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.

    :param pdf_file: The uploaded PDF file.
    :return: Extracted text as a string.
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    """
    Splits the text into chunks of a specified size.

    :param text: The text to be chunked.
    :param chunk_size: The size of each chunk.
    :return: A list of text chunks.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    """
    The main function that runs the application.
    """

    page_icon("💬")
    st.subheader("Ollama Playground", divider="red", anchor=False)

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
    else:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/03_⚙️_Settings.py")

    st.title("Chat with PDF")

    # PDF upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", pdf_text[:1000])  # Display first 1000 characters

        # Split the extracted text into smaller chunks
        text_chunks = chunk_text(pdf_text)

        # Chat interaction
        user_input = st.text_input("Ask something about the PDF:")
        if user_input:
            response = ""
            with st.spinner("Generating response from the model..."):
                for chunk in text_chunks:
                    # Use each chunk of the extracted text along with the user's input
                    messages = [
                        {"role": "system", "content": "The user has uploaded a PDF. The content is: " + chunk},
                        {"role": "user", "content": user_input}
                    ]

                    # Generate a response from the model for the current chunk
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        stream=True,
                    )

                    # Collect the response for each chunk
                    chunk_response = st.write_stream(stream)
                    response += chunk_response

                # Display the combined response from all chunks
                st.write("Assistant:", response)

    st.divider()

    message_container = st.container()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Model working..."):
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )
                # Stream response
                response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

        except Exception as e:
            st.error(e, icon="⛔️")


main()
