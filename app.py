import json
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


quiz_function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 {difficulty} questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Context: {context}
    """,
        )
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_format_doc, difficulty, topic):
    chain = (
        {
            "context": itemgetter("context"),
            "difficulty": itemgetter("difficulty"),
        }
        | prompt
        | llm
    )
    return chain.invoke({"context": _format_doc, "difficulty": difficulty})


with st.sidebar:
    st.write("https://github.com/seungchan0324/FullStackGPT_challenge_QuizGPT")
    key = st.text_input("Please give me your API Key!")
    docs = None
    topic = None
    if key:
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )

        difficulty = st.selectbox(
            "Choose your difficulty level!",
            [
                "easy",
                "medium",
                "challenging",
            ],
        )

        if choice == "File":
            file = st.file_uploader(
                "Upload a .docs, .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)

        else:
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini-2024-07-18",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
    api_key=key,
).bind(
    # function_call="auto" (자유롭게 function을 사용하도록 지시 할 때에.)
    function_call={"name": "create_quiz"},
    functions=[quiz_function],
)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.

    Before we begin, please choose your difficulty level!
    """
    )
else:
    format_doc = format_docs(docs)
    response = run_quiz_chain(format_doc, difficulty, topic)
    function_args_str = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(function_args_str)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong")
        button = st.form_submit_button()
