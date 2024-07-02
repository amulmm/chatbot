from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

st.title("Pharma-Chatbot")

load_dotenv()

pdf_path = "chatbot (1).pdf"

pdf_reader = PdfReader(pdf_path)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# split to chunks
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size =1000,
    chunk_overlap = 200,
    length_function = len
)
chunks = text_splitter.split_text(text)

# embedding
print("(INFO) Embedding")
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []

# chat logic
frage = st.chat_input("Enter your Question..")
if frage:
    docs = knowledge_base.similarity_search(frage)

    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=frage)
    st.session_state.messages.append({"role": "user", "content": frage})

    # bot message zu session state hinzufügen
    st.session_state.messages.append({"role": "assistant", "content": response})

# chat message zeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
