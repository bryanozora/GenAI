import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import fitz  # PyMuPDF for handling PDF content (pages, images)
import uuid  # For generating unique session IDs
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the Chatbot class
class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="bge-m3:latest", pdf_path=None):
        self.llm = llm
        self.embedding_model = embedding_model
        # Memory
        self.memory = self.create_memory()
        # Load the PDF document
        self.pdf_doc = self.load_pdf(pdf_path)
        # Ensure chat history is initialized
        self.chat_history = []

    def set_setting(self):
        # Check session state for a training adjustment, otherwise use default
        training_adjustment = st.session_state.get("training_adjustment", 1.0)

        # Set the settings dynamically for each prompt
        Settings.llm = Ollama(model=self.llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name=self.embedding_model)
        Settings.system_prompt = f"""
            You are a multi-lingual expert system who has knowledge based on real-time data.
            You will always try to be helpful and assist users in answering their questions. 
            If you don't know the answer, explicitly say that you DON'T KNOW, and DO NOT fabricate 
            answers outside the scope of the given document.

            Any training regimes you provide must account for the user's `training_adjustment` 
            factor, which is set to {training_adjustment:.2f}. Base your recommendations on this 
            adjustment factor to ensure they are appropriate for the user's fitness level and 
            goals. Clearly explain how the adjustment impacts the intensity, duration, and type 
            of training exercises you suggest.

            Your goal is to provide precise, personalized, and safe recommendations based on 
            the user's data, ensuring their well-being and progress.
        """

    @st.cache_resource(show_spinner=False)
    def load_data(_self, vector_store=None):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            index = VectorStoreIndex.from_documents(documents)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )

    # PDF handling methods
    def load_pdf(self, pdf_path):
        if pdf_path:
            return fitz.open(pdf_path)  # Load the PDF document
        return None

    def respond(self, prompt):
        # Refresh settings before each response
        self.set_setting()

        # Load or refresh index dynamically for each chat to use updated settings
        if not hasattr(self, "index"):
            self.index = self.load_data()
            self.chat_engine = self.create_chat_engine(self.index)

        # Generate chat response
        response = self.chat_engine.chat(prompt)
        return response.response

    def extract_keywords(self, response_text):
        # Extract potential keywords or exercise names (simple method using split)
        words = response_text.split()
        keywords = [word for word in words if len(word) > 3]  # Filter short, common words
        return keywords

    def load_exercises(self):
        # Load exact exercise names from each page of the PDF
        self.exercises = set()  # Use a set for faster look-up
        for i in range(len(self.pdf_doc)):
            page_text = self.pdf_doc.load_page(i).get_text("text")
            # Assuming exercises are listed line by line
            exercises_on_page = [line.strip() for line in page_text.splitlines() if line.strip()]
            self.exercises.update(exercises_on_page)

        # Compile regex patterns for exact matching exercises
        self.exercise_patterns = {
            exercise: re.compile(rf'\b{re.escape(exercise)}\b', re.IGNORECASE)
            for exercise in self.exercises
        }

    def extract_matching_exercises(self, response_text):
        # Identify exact matching exercises from bot response
        matching_exercises = set()
        for exercise, pattern in self.exercise_patterns.items():
            if pattern.search(response_text):
                matching_exercises.add(exercise)
        return matching_exercises
    
    def find_relevant_pages(self, response_text):
        if not self.pdf_doc:
            st.warning("PDF document not loaded.")
            return []

        # Load exercises and patterns if they haven't been loaded already
        if not hasattr(self, "exercises") or not self.exercises:
            self.load_exercises()

        # Get exact matching exercises from response
        matching_exercises = self.extract_matching_exercises(response_text)

        # Find pages containing any of the exact matched exercises
        relevant_pages = []
        for page_num in range(len(self.pdf_doc)):
            page_text = self.pdf_doc.load_page(page_num).get_text("text")
            if any(exercise in page_text for exercise in matching_exercises):
                relevant_pages.append(page_num)

        return relevant_pages

    def display_page_as_image(self, page_num):
        page = self.pdf_doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")  # Convert page to image bytes
        st.image(img_bytes, caption=f"Page {page_num + 1}", use_column_width=True)

    def respond_with_pdf_pages(self, response_text):
        relevant_pages = self.find_relevant_pages(response_text)
        if relevant_pages:
            for page_num in relevant_pages:
                self.display_page_as_image(page_num)

# Initialize chat session if not available 
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Initialize selected session if "selected_session" not in st.session_state:
if "selected_session" not in st.session_state:
    st.session_state.selected_session = None

# Function to create a new chat session with the default name
def create_new_session():
    session_id = f"session_{len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = {"name": "New Chat", "messages": []}
    st.session_state.selected_session = session_id

# Sidebar for managing chat sessions
with st.sidebar:
    st.write("Chat Sessions")

    # Check if there are no chat sessions and automatically create one
    if not st.session_state.chat_sessions:
        create_new_session()  # Automatically create a new session if none exist

    # Show list of sessions with session names being the last user message
    session_names = {session_id: session_data["name"] for session_id, session_data in st.session_state.chat_sessions.items()}
    selected_name = st.selectbox("Select a session", list(session_names.values()))

    # Get selected session ID based on name
    if selected_name:
        selected_session_id = list(st.session_state.chat_sessions.keys())[list(session_names.values()).index(selected_name)]
        st.session_state.selected_session = selected_session_id

    # Option to create a new session
    if st.button("Start New Session"):
        create_new_session()

# Main Program 
st.title("Gym Exercise Chatbot")

# Initialize the chatbot with your PDF file 
chatbot = Chatbot(pdf_path="photos/photos.pdf")  # Use the path to your uploaded PDF

# Display chat history of selected session
if st.session_state.selected_session:
    session_data = st.session_state.chat_sessions[st.session_state.selected_session]
    session_messages = session_data["messages"]

    for message in session_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chatbot.set_chat_history(session_messages)

    # React to user input in the selected session
    if prompt := st.chat_input("Ask something"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant's response and display it
        with st.chat_message("assistant"):
            response = chatbot.respond(prompt)
            st.markdown(response)

        # Display relevant PDF pages based on the bot's response
        chatbot.respond_with_pdf_pages(response)

        # After the bot's response, save both user prompt and bot's response
        session_messages.append({"role": "user", "content": prompt})
        session_messages.append({"role": "assistant", "content": response})

        # Update session name to be the last question asked by the user
        session_data["name"] = prompt[:50]  # Limit session name length to 50 characters
else:
    st.write("Please select or start a new chat session.")
