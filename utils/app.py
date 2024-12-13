import streamlit as st
from models import initialize_models, load_data
from retrieval import retrieve_context
from response_generator import generate_detailed_response
from prompts import create_prompt

# Initialize models and load data
@st.cache_resource
def init_models():
    return initialize_models()

@st.cache_resource
def load_all_data():
    return load_data()

# Streamlit app
def main():
    st.title("Ketto AI Chatbot")
    st.write("Hello! I'm Ketto, your friendly assistant for all things related to donations and social campaigns. Let's chat!")

    # Initialize models and data
    if "embedding_model" not in st.session_state:
        embedding_model, tokenizer, model = init_models()
        question_embeddings, data = load_all_data()
        st.session_state.embedding_model = embedding_model
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.question_embeddings = question_embeddings
        st.session_state.data = data

    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    # Step 1: Get the user's name
    user_name = st.text_input("May I know your name?", key="user_name")
    if user_name:
        st.write(f"Thank you, {user_name}! How can I assist you today?")

        # Step 2: Continuous single question-answer interaction
        query = st.text_area("Enter your question:", value="", key="query_input", placeholder="Type your question here...")
        if st.button("Send"):
            if query.strip():  # Ensure the query is not empty
                with st.spinner("Retrieving context and generating response..."):
                    # Retrieve context and generate response
                    context = retrieve_context(
                        query.strip(),
                        st.session_state.question_embeddings,
                        st.session_state.data,
                        st.session_state.embedding_model
                    )
                    response = generate_detailed_response(
                        context,
                        query.strip(),
                        st.session_state.tokenizer,
                        st.session_state.model,
                        lambda ctx, q: create_prompt(ctx, q, user_name=user_name)
                    )

                # Store the last response
                st.session_state.last_response = response

        # Display only the last response
        if st.session_state.last_response:
            st.subheader("Ketto's Response:")
            st.write(st.session_state.last_response)

if __name__ == "__main__":
    main()
