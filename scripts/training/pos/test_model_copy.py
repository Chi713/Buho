import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def assemble(tokens: list[str], predictions: list[str], answers: list[str]):
    tokens.reverse()
    predictions.reverse()
    answers.reverse()

    assembled_tokens = []
    new_predictions = []
    new_answers = []
    temp_token = ""    
    for token, prediction, answer in zip(tokens, predictions, answers):
        if token.startswith('##'):
            temp_token = token[2:] + temp_token
        else:
            token = token + temp_token
            assembled_tokens.append(token)
            new_predictions.append(prediction)
            new_answers.append(answer)
            # print(f"token: {token}")
            temp_token = ""
    assembled_tokens.reverse()
    new_predictions.reverse()
    new_answers.reverse()

    # print(assembled_tokens, new_predictions, new_answers)
    return assembled_tokens, new_predictions, new_answers

# Environment variables for paths
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')

POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")
TEST_PATH = os.path.join(DATA_PATH, "test_inputs_pos.pt")

# Load the saved test data (assumes it has keys: 'input_ids', 'attention_mask', 'labels')
test_data = torch.load(TEST_PATH)  
input_ids = test_data["input_ids"]          # Tensor of shape [N, seq_length]
attention_mask = test_data["attention_mask"]# Tensor of shape [N, seq_length]
labels = test_data["labels"]                # Tensor of shape [N, seq_length]
# print(f"labels: {labels.shape}")

# Define the POS label mappings
POS_TO_ID = {
    '': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6,
    'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 'PRON': 11, 'PROPN': 12,
    'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '_': 18
}
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained(POS_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
model.eval()

# Move to GPU if available
# device = torch.device("cpu") # only temp cuz perlmutter
# model.to(device)
# input_ids = input_ids.to(device)
# attention_mask = attention_mask.to(device)
# labels = labels.to(device)

# Run inference under torch.no_grad() to disable gradient calculations
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

total_token = 0
total_correct = 0

# Evaluate accuracy
# We assume that the data has special tokens (like [CLS], [SEP]) handled similarly.
# print(f"prediction: {predictions.shape}")

for input_seq, pred_seq, label_seq in zip(input_ids, predictions, labels):
    input_seq =  tokenizer.convert_ids_to_tokens(input_seq)
    pred_seq = [pred_seq[i].item() for i in range(len(input_seq))]
    label_seq = [label_seq[i].item() for i in range(len(input_seq))]
    tokens, predictions, answers = assemble(input_seq, pred_seq, label_seq)
    for t, p, a in zip(tokens, predictions, answers):
        # If your dataset uses -100 or another special label to mark padding/ignored positions, skip them:
        if a == 0:
            continue
        total_token += 1
        if p == a:
            total_correct += 1
        if total_token % 100 == 0:
            print(f"correct {total_correct} out of {total_token} tokens")

accuracy = total_correct / total_token if total_token > 0 else 0.0
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
