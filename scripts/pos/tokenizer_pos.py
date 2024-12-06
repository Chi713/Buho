import os
import json
import torch
from transformers import AutoTokenizer

# define model path
MODELS_PATH = os.environ.get('BUHO_MODELS_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')
POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
# tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Paths to your data files
DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
TRAIN_FILE = os.path.join(DATA_PATH, "train_data.json")
DEV_FILE = os.path.join(DATA_PATH, "dev_data.json")
TEST_FILE = os.path.join(DATA_PATH, "test_data.json")

# Load your data
def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Tokenize and align labels
def tokenize_and_align_labels(sentences, pos_tags, max_length=256):
    """
    Tokenizes sentences and aligns POS tags with subword tokens.
    
    Parameters:
    - sentences: List of sentences (each sentence is a list of words).
    - pos_tags: List of POS tag IDs corresponding to each word in the sentences.
    - max_length: Maximum sequence length for padding/truncation.

    Returns:
    - A dictionary containing tokenized input IDs, attention masks, and aligned labels.
    """
    inputs = tokenizer(
        sentences,
        is_split_into_words=True,  # Treat each sentence as pre-tokenized
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",  # Return PyTorch tensors
    )

    labels = []
    for i, sentence in enumerate(sentences):
        word_ids = inputs.word_ids(batch_index=i)  # Map subword tokens to original words
        label_ids = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
            # Special tokens like [CLS], [SEP], or padding
                label_ids.append(0)
            else:
            # Check the next token to see if it belongs to the same word
                if idx < len(word_ids) - 1 and word_ids[idx + 1] == word_idx:
                    # Another subword for the same word is coming next, so this is NOT the last subword
                    label_ids.append(18)  # Ignored subword token
                else:
                # This is the last subword of the word (or the only one if it's a single-subword word)
                    label_ids.append(pos_tags[i][word_idx])

        labels.append(label_ids)
    inputs["labels"] = torch.tensor(labels)
    return inputs


# Process datasets
def process_datasets(train_file, dev_file, test_file):
    """
    Tokenizes and aligns labels for train, dev, and test datasets.
    
    Returns:
    - Tokenized PyTorch datasets for train, dev, and test splits.
    """
    train_data = load_data(train_file)
    dev_data = load_data(dev_file)
    test_data = load_data(test_file)

    train_inputs = tokenize_and_align_labels(train_data["sentences"], train_data["pos_tags"])
    dev_inputs = tokenize_and_align_labels(dev_data["sentences"], dev_data["pos_tags"])
    test_inputs = tokenize_and_align_labels(test_data["sentences"], test_data["pos_tags"])

    return train_inputs, dev_inputs, test_inputs

# Save processed data
def save_tokenized_data(tokenized_data, output_file):
    """Save tokenized data to a PyTorch file."""
    torch.save(tokenized_data, output_file)
    print(f"Saved tokenized data to {output_file}.")

# Main execution
if __name__ == "__main__":
    # Process and save the datasets
    train_inputs, dev_inputs, test_inputs = process_datasets(TRAIN_FILE, DEV_FILE, TEST_FILE)

    save_tokenized_data(train_inputs, os.path.join(DATA_PATH, "train_inputs_pos.pt"))
    save_tokenized_data(dev_inputs, os.path.join(DATA_PATH, "dev_inputs_pos.pt"))
    save_tokenized_data(test_inputs, os.path.join(DATA_PATH, "test_inputs_pos.pt"))
