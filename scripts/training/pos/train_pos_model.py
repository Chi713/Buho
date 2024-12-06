import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    AutoModel,
)
from sklearn.metrics import classification_report
import numpy as np
# from torchcrf import CRF


# Paths to tokenized data
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH')
POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")
PRETRAINED_MODEL = os.environ.get('PRETRAINED_MODEL')

TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_pos.pt")
DEV_PATH = os.path.join(DATA_PATH, "dev_inputs_pos.pt")
TEST_PATH = os.path.join(DATA_PATH, "test_inputs_pos.pt")

# POS Mapping
POS_TO_ID = {'-PAD-': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
             'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '_': 18 }
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

# class BertWithCRF(torch.nn.Module):
#     def __init__(self, model_name, num_labels):
#         super(BertWithCRF, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_name)  # BERT backbone
#         self.dropout = torch.nn.Dropout(0.1)  # Dropout for regularization
#         self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)  # Emission scores
#         self.crf = CRF(num_labels, batch_first=True)  # CRF layer

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         logits = self.classifier(self.dropout(outputs.last_hidden_state))  # Emission scores
#         if labels is not None:
#             # Compute the CRF loss during training
#             loss = -self.crf(logits, labels, mask=attention_mask.byte())
#             return loss
#         else:
#             # Decode the best label sequence during inference
#             predictions = self.crf.decode(logits, mask=attention_mask.byte())
#             return predictions

# Dataset Class
class POSTaggingDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data["input_ids"])
    def __getitem__(self, idx):
        data = {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }
        # Debug: Print unique label values
        if idx == 0:  # Print once
            print(f"Unique labels in the dataset: {set(data['labels'].tolist())}")
        return data

# Load Datasets
train_dataset = POSTaggingDataset(TRAIN_PATH)
dev_dataset = POSTaggingDataset(DEV_PATH)
test_dataset = POSTaggingDataset(TEST_PATH)

# Load BERT Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(
    PRETRAINED_MODEL,
    num_labels=len(POS_TO_ID)  # Set the number of labels
)

model.resize_token_embeddings(len(tokenizer))

# Data Collator (Handles Padding Dynamically During Training)
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training Arguments
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",  # Replace evaluation_strategy with eval_strategy
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=16,
    weight_decay=0.01,
    logging_dir="../logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    save_strategy="epoch",
    gradient_accumulation_steps=2,
    fp16=True,
    logging_steps=10,
)


# Trainer Object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train Model
trainer.train()

# Evaluate Model
predictions, labels, _ = trainer.predict(test_dataset)
print(predictions.shape)
print(predictions)
predictions = np.argmax(predictions, axis=2)

# Align Predictions and Labels
y_pred = []
y_true = []

for pred, label in zip(predictions, labels):
    for p, l in zip(pred, label):
        if l != 0:  # Ignore padding
            y_pred.append(p)
            y_true.append(l)
            

# Dynamically include only the labels present in the test set
unique_labels = sorted(set(y_true))  # Ensure unique labels in the test set
filtered_target_names = [ID_TO_POS[i] for i in unique_labels]

print(classification_report(
    y_true,
    y_pred,
    target_names=filtered_target_names,
    labels=unique_labels  # Use only the labels present in the test set
))

# Save Model
model.save_pretrained(POS_MODEL_PATH)
print(f"Model and tokenizer saved to '{MODEL_PATH}'.")