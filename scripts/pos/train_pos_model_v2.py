import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
from torchcrf import CRF  # Conditional Random Field
import numpy as np
from transformers import (DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments)
from huggingface_hub import ModelHubMixin

# Paths to tokenized data
DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH') + "_v2"
POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")
PRETRAINED_MODEL = os.environ.get('PRETRAINED_MODEL')

TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_pos.pt")
DEV_PATH = os.path.join(DATA_PATH, "dev_inputs_pos.pt")
TEST_PATH = os.path.join(DATA_PATH, "test_inputs_pos.pt")

# POS Mapping
POS_TO_ID = {'': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
             'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '_': 18 }
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

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
        # Debugging: Check for invalid labels
        if idx == 0:  # Only print for the first sample
            print(f"Input IDs (sample {idx}): {data['input_ids']}")
            print(f"Attention Mask (sample {idx}): {data['attention_mask']}")
            print(f"Labels (sample {idx}): {data['labels']}")
            print(f"Max Label: {data['labels'].max()}, Min Label: {data['labels'].min()}")
        return data


# Load Datasets

train_dataset = POSTaggingDataset(TRAIN_PATH)
dev_dataset = POSTaggingDataset(DEV_PATH)
test_dataset = POSTaggingDataset(TEST_PATH)

# Load BERT Model with CRF
class BertWithCRF(torch.nn.Module, ModelHubMixin):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL)  # BERT backbone
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
        self.bert.resize_token_embeddings(len(tokenizer))  # resize the original BERT module to custom tokenizer
        self.dropout = torch.nn.Dropout(0.1)  # Dropout for regularization
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)  # Emission scores
        print(f"num_labels: {num_labels}")
        self.crf = CRF(num_labels, batch_first=True)  # CRF layer

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))  # Emission scores
        
        # Debugging: Log shapes and max values
        # print(f"\n[Debug] Logits Shape: {logits.shape} (batch_size, seq_len, num_labels)")
        # if labels is not None:
            # print(f"[Debug] Labels Shape: {labels.shape}")
            # print(f"[Debug] Max Label: {labels.max()}, Min Label: {labels.min()}")
            # print(f"[Debug] Attention Mask Shape: {attention_mask.shape}")

        if logits.size(-1) != len(POS_TO_ID):
            print(f"[Error] Mismatch: Logits last dimension ({logits.size(-1)}) != num_labels ({len(POS_TO_ID)})")

        if labels is not None:
            # Ensure all labels are valid
            invalid_labels = labels[labels >= logits.size(-1)].tolist()
            if invalid_labels:
                print(f"[Error] Invalid Labels Detected: {invalid_labels}")

            loss = -self.crf(logits,tags=labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
model = BertWithCRF(num_labels=(len(POS_TO_ID)))
# check crf init
print(f"crf init: {model.crf.transitions}")

# Data Collator (Handles Padding Dynamically During Training)
data_collator = DataCollatorForTokenClassification(AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH), padding=True)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)


# Training Arguments
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",  # Replace evaluation_strategy with eval_strategy
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
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


# # Optimizer
# optimizer = AdamW(model.parameters(), lr=5e-5)

# # Training Loop
# num_epochs = 5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         # Debugging: Check batch stats
#         # if step == 0:  # Print only for the first batch
#             # print(f"[Debug] Step {step}: Input IDs Shape: {input_ids.shape}")
#             # print(f"[Debug] Step {step}: Attention Mask Shape: {attention_mask.shape}")
#             # print(f"[Debug] Step {step}: Labels Shape: {labels.shape}")
#             # print(f"[Debug] Step {step}: Max Label: {labels.max()}, Min Label: {labels.min()}")

#         optimizer.zero_grad()
#         try:
#             loss = model.forward(input_ids, attention_mask, labels)  # Compute CRF loss
#             print('loss:', loss)
#         except RuntimeError as e:
#             print(f"[Error] RuntimeError in Step {step}: {e}")
#             print(f"[Debug] Input IDs: {input_ids}")
#             print(f"[Debug] Attention Mask: {attention_mask}")
#             print(f"[Debug] Labels: {labels}")
#             raise e

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

# # Evaluation

# # Validation
# model.eval()
# predictions, true_labels = [], []
# with torch.no_grad():
#     for step, batch in enumerate(tqdm(dev_loader, desc="Validating")):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         # Debugging: Check batch stats
#         print(f"\n[Step {step}]")
#         # print(f"Input IDs Shape: {input_ids.shape}")
#         # print(f"Attention Mask Shape: {attention_mask.shape}")
#         # print(f"Labels Shape: {labels.shape}")
#         # print(f"Max Label: {labels.max()}, Min Label: {labels.min()}")

#         try:
#             # Pass through the model
#             preds = model(input_ids, attention_mask)  # Decode CRF predictions

#             # Debugging: Log predictions
#             # print(f"[Debug] Predictions Length: {len(preds)}")
#             # print(f"[Debug] First Prediction (Example): {preds[0]}")

#         except RuntimeError as e:
#             print(f"[Error] RuntimeError during validation at Step {step}: {e}")
#             print(f"[Debug] Input IDs: {input_ids}")
#             print(f"[Debug] Attention Mask: {attention_mask}")
#             print(f"[Debug] Labels: {labels}")
#             raise e

#         # Extend results
#         predictions.extend(preds)
#         true_labels.extend(labels.cpu().tolist())


# # Flatten predictions and labels
# y_pred, y_true = [], []
# for pred, label in zip(predictions, true_labels):
#     for p, l in zip(pred, label):
#         if l != -100:  # Ignore padding
#             y_pred.append(p)
#             y_true.append(l)

# # Map predictions and labels to human-readable POS tags
# y_pred_labels = [ID_TO_POS[i] for i in y_pred]
# y_true_labels = [ID_TO_POS[i] for i in y_true]

# # Print classification report
# print(classification_report(y_true_labels, y_pred_labels))


# Evaluate Model
predictions, labels, trial = trainer.predict(test_dataset)
print(predictions.shape)
print(predictions)
print(labels.shape)
print(labels)
print(trial)
print(trial.shape)
# predictions = np.argmax(predictions)
# predictions = predictions.argmax()

# Align Predictions and Labels
y_pred = []
y_true = []

for pred, label in zip(predictions, labels):
    y_pred.extend(pred)
    y_true.extend(label[label != 0].to_list())
    # for p, l in zip(pred, label):
    #     if l != 0:  # Ignore padding
    #         y_pred.append(p)
    #         y_true.append(l)
            

# Dynamically include only the labels present in the test set
unique_labels = sorted(set(y_true))  # Ensure unique labels in the test set
filtered_target_names = [ID_TO_POS[i] for i in unique_labels]

print(classification_report(
    y_true,
    y_pred,
    target_names=filtered_target_names,
    labels=unique_labels  # Use only the labels present in the test set
))

# Save Model and Tokenizer
model.save_pretrained(POS_MODEL_PATH)
print(f"Model saved to {POS_MODEL_PATH}.")
