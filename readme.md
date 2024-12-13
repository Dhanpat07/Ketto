
# Ketto AI Chatbot

Ketto AI Chatbot is a Streamlit-based conversational AI application designed to assist users with donation and social campaign inquiries. Powered by the Falcon 7B open-source model, it provides accurate and structured responses based on relevant context.

---

## Features

- **Streamlit UI**: User-friendly interface for seamless interaction.
- **Context Retrieval**: Retrieves the most relevant information using cosine similarity.
- **Structured Responses**: Generates clear, concise, and informative answers.
- **Efficient Query Handling**: Reduces hallucination by grounding answers in the retrieved context.
- **Open-Source AI**: Utilizes Falcon 7B model for natural language understanding and response generation.

---

## Project Structure

```
.
├── app.py                   # Main Streamlit application for UI
├── context_embeddings.py    # Creates embeddings for structured Q&A data
├── download_model.py        # Downloads the Falcon model and tokenizer
├── models.py                # Initializes and loads the Falcon model and tokenizer
├── prompts.py               # Defines instructions for generating structured answers
├── response_generator.py    # Generates responses using the Falcon model and context
├── retrieval.py             # Retrieves relevant documents using cosine similarity
├── requirements.txt         # List of dependencies for the project
├── data/
│   ├── raw_data.json        # Raw data for creating Q&A format
│   ├── questions.json       # Structured Q&A data
│   └── embeddings.pkl       # Precomputed embeddings for questions
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ketto-ai-chatbot.git
   cd ketto-ai-chatbot
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.8 or higher installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Falcon 7B Model**:
   Use the `download_model.py` script to download the Falcon model and tokenizer:
   ```bash
   python download_model.py
   ```

4. **Generate Context Embeddings**:
   After structuring the Q&A data, use `context_embeddings.py` to create embeddings:
   ```bash
   python context_embeddings.py
   ```

---

## Usage

1. **Run the Application**:
   Start the Streamlit UI:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the Chatbot**:
   - Open the URL provided by Streamlit (e.g., `http://localhost:8501`).
   - Provide your name and start asking questions.

---

## File Descriptions

### 1. `context_embeddings.py`
- **Purpose**: Processes structured Q&A data to generate embeddings.
- **Output**: Stores embeddings in `data/embeddings.pkl` for efficient context retrieval.

### 2. `download_model.py`
- **Purpose**: Downloads the Falcon 7B model and tokenizer required for response generation.

### 3. `models.py`
- **Purpose**: Initializes and loads the Falcon 7B model and tokenizer.

### 4. `retrieval.py`
- **Purpose**: Retrieves the most relevant document for a query using cosine similarity.

### 5. `prompts.py`
- **Purpose**: Defines structured prompts to guide the model in generating accurate responses.

### 6. `response_generator.py`
- **Purpose**: Uses retrieved context and Falcon 7B to generate concise and grounded responses.

### 7. `app.py`
- **Purpose**: Provides a Streamlit-based user interface for interacting with the chatbot.

### 8. `requirements.txt`
- **Purpose**: Specifies the dependencies required for the project.

---

## Workflow

1. **Prepare Data**:
   - Structure raw data into Q&A format (`data/raw_data.json` -> `data/questions.json`).

2. **Generate Embeddings**:
   - Use `context_embeddings.py` to create embeddings from the structured data.

3. **Download Model**:
   - Run `download_model.py` to download Falcon 7B and its tokenizer.

4. **Run Chatbot**:
   - Launch the Streamlit app using `app.py`.

5. **Ask Questions**:
   - Interact with the chatbot and receive structured responses.

---

## Future Enhancements

- **Multilingual Support**: Extend support to multiple languages.
- **Real-Time Updates**: Integrate live data sources for dynamic responses.
- **Voice Interaction**: Add speech-to-text and text-to-speech capabilities.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For inquiries or collaboration:

- **Name**: Dhanpat
- **Email**: dhanpat@example.com

Happy Coding! 🚀
