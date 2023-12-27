import warnings
import streamlit as st
from streamlit_chat import message as st_message
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains import RetrievalQA
from PIL import Image
import base64
# Set environment variables
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyBaDRZuLLst7ChknpcA9hS1nmiVdP4aNZI"
DB_FAISS_PATH = "VectorStore/db_faiss"

# Page title
st.title("College-Info 👨🏽‍💻")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# error:

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     llm = GooglePalm(temperature=0.1)


# Function to embed and update the index
def embed_index(doc_list, embed_fn, index_store):

    if os.path.exists(index_store):
        local_db = FAISS.load_local(index_store, embed_fn)
        # local_db.merge_from(faiss_db)
        # print("Merge completed")
        # local_db.save_local(index_store)
        # print("Updated index saved")
    else:
        try:
            faiss_db = FAISS.from_documents(doc_list, embed_fn)
        except Exception as e:
            faiss_db = FAISS.from_texts(doc_list, embed_fn)
        faiss_db.save_local(folder_path=index_store)
        # print("New store created...")


# Load data and create models
@st.cache_resource
def get_models():
    loader = CSVLoader(file_path="Data/dataset.csv",
                       encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    embeddings = GooglePalmEmbeddings()

    # Specify the path where you want to save/load the vectorstore
    index_store_path = "VectorStore2"
    # os.makedirs(index_store_path, exist_ok=True)

    # Use the embed_index function to update or create the vectorstore
    # embed_index(text_chunks, embeddings, index_store_path)

    vectorstore = FAISS.load_local(index_store_path, embeddings)

    llm = GooglePalm(temperature=0.1)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    return qa, vectorstore


qa_model, vectorstore = get_models()

# ... (rest of the code)


# Generate answers and update chat history


def generate_answer():
    user_message = st.session_state.input_text

    # Check if the user has entered a message
    if user_message:
        answer = qa_model.run(user_message)

        # Append user message and bot answer to chat history
        st.session_state.history.append(
            {"message": user_message, "is_user": True})
        st.session_state.history.append({"message": answer, "is_user": False})


# Main chatbot input and output
st.subheader("Ask me anything about college!")
user_message = st.text_input("Your Question:", key="input_text",
                             on_change=generate_answer, help="Type your question and press Enter.")
st.text_area("Answer:", value=st.session_state.history[-1]["message"]
             if st.session_state.history else "", height=100, max_chars=None, disabled=True)

# Buttons for additional functionality
if st.button("Clear History"):
    st.session_state.history = []

# Image in the sidebar
# Image in the sidebar with auto-adjusted width and height
# Display image with automatic width adjustment using PIL
# image_path = "C:/Users/pankaj/Documents/college1.png"
# image = Image.open(image_path)
# # st.sidebar.image(image, width=None,
#                  caption="Hey , Welcome to our chatbot!")

# image sidebar:
sidebar_header_style = """
<style>

[data-testid="stSidebar"] div:first-child div div h2  {
    font-family: cursive, sans-serif;  
    font-size: 20px; 
    margin-top: auto;
    margin-bottom: 0; 
    color: red !important;  
    cursor: pointer;  
</style>
"""
# Apply the custom style to the sidebar header
st.markdown(sidebar_header_style, unsafe_allow_html=True)

st.sidebar.header("Let's Get You A College 🎓!")


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb")as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("Images/college1.png")


page_bg_img = f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    background-image: url('data:image/png;base64,{img}');
    background-size: cover;
    background-position: {65}% {50}%;
    background-clip: content-box;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Header for chat history
st.subheader("Chat History!")

for chat in st.session_state.history:
    st_message(**chat)  # unpacking
