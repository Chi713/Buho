import os
import json
from transformers import BertTokenizer
import re

# Paths to your JSON files
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
TRAIN_FILE = os.path.join(DATA_PATH, "train_data.json")
DEV_FILE = os.path.join(DATA_PATH, "dev_data.json")
TEST_FILE = os.path.join(DATA_PATH, "test_data.json")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def load_data(file_path):
    """Loads the JSON data file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_valid_word(word):
    """
    Checks if the word is valid:
    - No commas, periods, or numeric characters.
    """
    if any(char in word for char in [",", ".", "!", "?"]):  # Filter punctuation
        return False
    if any(char.isdigit() for char in word):  # Filter numeric characters
        return False
    return True

def find_fragmented_words(data):
    """Find words that are fragmented by the tokenizer."""
    fragmented_words = set()
    
    for sentence in data["sentences"]:
        for word in sentence:
            if is_valid_word(word):  # Filter out invalid words
                tokenized = tokenizer.tokenize(word)
                if len(tokenized) > 1:  # If the word is split into multiple tokens
                    fragmented_words.add(word)
    
    return fragmented_words

# Process all datasets and combine fragmented words
all_fragmented_words = set()

for file_path in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
    data = load_data(file_path)
    fragmented_words = find_fragmented_words(data)
    all_fragmented_words.update(fragmented_words)

# Save all fragmented words to a single file
output_file = os.path.join(DATA_PATH, "all_filtered_fragmented_words.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(all_fragmented_words)))  # Sort for consistency

print(f"Filtered fragmented words saved to {output_file}")
