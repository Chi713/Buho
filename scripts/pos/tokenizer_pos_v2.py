import os
import json
import torch
from transformers import AutoTokenizer

# Define model path
MODELS_PATH = os.environ.get('BUHO_MODELS_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')+'_v2'
POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")

POS_TO_ID = {'':0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
             'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '_': 18 }
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)

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

# Inspect tokenized data
def inspect_sample(tokenized_data, tokenizer, sample_idx=0):
    """Print a sample from the tokenized dataset for manual inspection."""
    print("\nInspecting Sample:")
    print(f"Input IDs: {tokenized_data['input_ids'][sample_idx]}")
    print(f"Attention Mask: {tokenized_data['attention_mask'][sample_idx]}")
    print(f"Labels: {tokenized_data['labels'][sample_idx]}")

    # Decode input IDs to verify tokenization
    tokens = tokenizer.convert_ids_to_tokens(tokenized_data['input_ids'][sample_idx])
    print(f"Tokens: {tokens}")

    # Pair tokens with labels for clarity
    labels = tokenized_data['labels'][sample_idx].tolist()
    token_label_pairs = [(token, label) for token, label in zip(tokens, labels) if label != -100]
    print(f"Token-Label Pairs: {token_label_pairs}")

# Tokenize and align labels
def tokenize_and_align_labels(sentences, pos_tags, max_length=256):
    """
    Tokenizes sentences and aligns POS tags with subword tokens.
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
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx == previous_word_idx:
                # Set -100 subword tokens (ignored during loss computation)
                label_ids.append(17) #pos_tag 
            else:
                try:
                    label_ids.append(pos_tags[i][word_idx])  # Align POS tag to the original word
                except IndexError:
                    print(f"[Error] IndexError for sentence {i}, word index {word_idx}")
                    print(f"Sentence: {sentence}")
                    print(f"POS Tags: {pos_tags[i]}")
                    raise

            previous_word_idx = word_idx

        # Validate label IDs
        if max(label_ids) >= len(POS_TO_ID):
            print(f"[Error] Invalid label value detected in sentence {i}. Max label: {max(label_ids)}")
            print(f"Sentence: {sentence}")
            print(f"POS Tags: {pos_tags[i]}")
            raise ValueError("Invalid label value found during alignment.")

        labels.append(label_ids)

        # Check alignment
        valid_labels = [label for label in label_ids if label != -100]
        if len(valid_labels) != len(sentence):
            print(f"[Warning] Mismatch in token-label alignment for sentence {i}.")
            print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])}")
            print(f"Labels: {label_ids}")
            print(f"Sentence: {sentence}")
            print(f"POS Tags: {pos_tags[i]}")

    # Add aligned labels to inputs
    inputs["labels"] = torch.tensor(labels)

    # Debugging: Limit outputs to 3 sentences
    for i in range(min(3, len(sentences))):  # Only print for the first 3 sentences
        word_ids = inputs.word_ids(batch_index=i)
        print(f"\nDebug Sentence {i}:")
        print(f"Word IDs: {word_ids}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])}")
        print(f"Labels: {labels[i]}")
        print(f"Original Sentence: {sentences[i]}")
        print(f"Original POS Tags: {pos_tags[i]}")

    return inputs


# Process datasets
def process_datasets(train_file, dev_file, test_file):
    """
    Tokenizes and aligns labels for train, dev, and test datasets.
    """
    train_data = load_data(train_file)
    dev_data = load_data(dev_file)
    test_data = load_data(test_file)

    # Validate input data
    for split_name, data in zip(["train", "dev", "test"], [train_data, dev_data, test_data]):
        for i, (sentence, pos_tags) in enumerate(zip(data["sentences"], data["pos_tags"])):
            if len(sentence) != len(pos_tags):
                print(f"[Error] Mismatched sentence and POS tag lengths in {split_name} data.")
                print(f"Sentence {i}: {sentence}")
                print(f"POS Tags {i}: {pos_tags}")
                raise ValueError(f"Length mismatch in {split_name} data at index {i}.")

    train_inputs = tokenize_and_align_labels(train_data["sentences"], train_data["pos_tags"])
    dev_inputs = tokenize_and_align_labels(dev_data["sentences"], dev_data["pos_tags"])
    test_inputs = tokenize_and_align_labels(test_data["sentences"], test_data["pos_tags"])

    return train_inputs, dev_inputs, test_inputs

# Save processed data
def save_tokenized_data(tokenized_data, output_file):
    """Save tokenized data to a PyTorch file and validate the data."""
    print(f"Saving tokenized data to {output_file}...")
    print(f"Input IDs Shape: {tokenized_data['input_ids'].shape}")
    print(f"Attention Mask Shape: {tokenized_data['attention_mask'].shape}")
    print(f"Labels Shape: {tokenized_data['labels'].shape}")
    print(f"Max Label: {tokenized_data['labels'].max()}, Min Label: {tokenized_data['labels'].min()}")

    # Ensure no invalid labels
    invalid_labels = tokenized_data["labels"][(tokenized_data["labels"] != -100) & 
                                              (tokenized_data["labels"] >= len(POS_TO_ID))]
    if invalid_labels.numel() > 0:
        print(f"[Error] Found invalid labels in the tokenized data: {invalid_labels}")
        raise ValueError("Invalid labels detected in tokenized data.")

    torch.save(tokenized_data, output_file)
    print(f"Tokenized data saved successfully.")

# Main execution
if __name__ == "__main__":
    # Process and save the datasets
    train_inputs, dev_inputs, test_inputs = process_datasets(TRAIN_FILE, DEV_FILE, TEST_FILE)

    save_tokenized_data(train_inputs, os.path.join(DATA_PATH, "train_inputs_pos.pt"))
    save_tokenized_data(dev_inputs, os.path.join(DATA_PATH, "dev_inputs_pos.pt"))
    save_tokenized_data(test_inputs, os.path.join(DATA_PATH, "test_inputs_pos.pt"))

    # Inspect samples
    inspect_sample(train_inputs, tokenizer, sample_idx=0)
    inspect_sample(dev_inputs, tokenizer, sample_idx=0)
    inspect_sample(test_inputs, tokenizer, sample_idx=0)
