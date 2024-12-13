from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

def initialize_models():
    """Initialize embedding and Falcon models."""
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading Falcon model and tokenizer...")
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    return embedding_model, tokenizer, model

def load_data():
    """Load pre-saved embeddings and Q&A data."""
    with open("data/raw/question_embeddings.pkl", "rb") as f:
        question_embeddings = pickle.load(f)

    with open("data/raw/qa_data.pkl", "rb") as f:
        data = pickle.load(f)

    return question_embeddings, data
