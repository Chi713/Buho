import os
from transformers import BertTokenizer
from transformers import BertForMaskedLM
# Path to the fragmented words file
MODELS_PATH = os.environ.get('BUHO_MODELS_PATH')
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODEL_NAME = os.environ.get('BUHO_MODEL_NAME')
OUTPUT_MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
FRAGMENTED_WORDS_FILE = os.path.join(DATA_PATH, "all_filtered_fragmented_words.txt")

# Load the existing tokenizer
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Read the fragmented words from the file
with open(FRAGMENTED_WORDS_FILE, "r", encoding="utf-8") as f:
    fragmented_words = f.read().splitlines()

# Print the number of words to be added
print(f"Number of new tokens to add: {len(fragmented_words)}")

# Add new tokens to the tokenizer
num_added_tokens = tokenizer.add_tokens(fragmented_words)
print(f"Added {num_added_tokens} tokens to the tokenizer.")

# Save the updated tokenizer
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print(f"Updated tokenizer saved to {OUTPUT_MODEL_PATH}")

# Load the pre-trained model
model = BertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Resize the embeddings to match the updated tokenizer
model.resize_token_embeddings(len(tokenizer))

# Save the updated model
model.save_pretrained(OUTPUT_MODEL_PATH)
print(f"Updated model saved to {OUTPUT_MODEL_PATH}")
