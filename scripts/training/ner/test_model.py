import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import time
import json
from scripts.utils.fragment_assembler import assemble

DATA_PATH = os.environ.get('BUHO_DATA_PATH')  # Replace with the actual dataset path
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH') #+ '_v2'
NER_MODEL_PATH = os.path.join(MODEL_PATH, "ner")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")

TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_ner.pt")
TEST_FILE = os.path.join(DATA_PATH, "test_ner_data.json")
def load_data(file_path):
    """Loads the JSON data file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data(TEST_FILE)

sentence = [ ner_tags_list for ner_tags_list in data['sentences']][0]
answers = [ ner_tags_list for ner_tags_list in data['ner_tags']][0]
# Define ID_TO_NER mapping
NER_TO_ID = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ID_TO_NER = {v: k for k, v in NER_TO_ID.items()}


# Load model and tokenizer
start_time = time.time()

model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
model.eval()
load_time = time.time()

# Tokenize input with `is_split_into_words`
print(sentence)
print(len(answers))
inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", truncation=True)
print(f"answers tokenized")

labels = []
word_ids = inputs.word_ids()  # Map subword tokens to original words
label_ids = []
previous_word_idx = None

for word_idx in word_ids:
    if word_idx is None:
        label_ids.append(0)
    elif word_idx == previous_word_idx:
        # Set -100 for padding and subword tokens (ignored during loss computation)
        label_ids.append(0)
    else:
        label_ids.append(answers[word_idx])  # Align NER tag to the original word
    previous_word_idx = word_idx

labels.append(label_ids)
# answers = [ insert_return(ner_tags_list) for ner_tags_list in labels][0]
# print(answers)

tokenize_time = time.time()

# Run inference
outputs = model(**inputs)
tagging_time = time.time()

# Decode predictions
init_predictions = torch.argmax(outputs.logits, dim=-1)[0]
split_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # Convert input IDs to tokens
predicted_tags = [init_predictions[i].item() for i in range(len(split_tokens))]

tokens = split_tokens
predictions = predicted_tags
print(predictions)
print(len(predictions))
print(len(labels[0]))

total_token=0
total_correct=0

# Display the results
tokens, predictions, answers = assemble(tokens, predictions, labels[0])
for token, tag, answer in zip(tokens, predictions, answers):
    # Skip special tokens
    if token in ["[CLS]", "[SEP]"]:
        continue
    print(f"{token}: {ID_TO_NER.get(tag, 'UNKNOWN')}: {answer}")
    total_token += 1
    total_correct += tag == answer

percent_correct = total_correct/total_token

print(f"load time: {load_time - start_time}")
print(f"tokenizing time: {tokenize_time - load_time}")
print(f"tagging time: {tagging_time - tokenize_time}")
print(f"total time: {tagging_time - start_time}")
print(f"Percentage correct: {percent_correct}")