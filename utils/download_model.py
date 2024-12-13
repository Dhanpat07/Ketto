
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pre-download and cache Falcon-7B-Instruct model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
cache_dir = "./falcon_model_cache"

# Download tokenizer and model
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

print("Model and tokenizer downloaded and cached locally in:", cache_dir)
