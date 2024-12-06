import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm
from torchcrf import CRF  # Conditional Random Field
import numpy as np
from transformers import DataCollatorForTokenClassification
import deepspeed  # Import DeepSpeed


DATA_PATH = os.environ.get('BUHO_DATA_PATH')
MODEL_PATH = os.environ.get('BUHO_MODEL_PATH') + "_v3"
POS_MODEL_PATH = os.path.join(MODEL_PATH, "pos")
TOKENIZER_MODEL_PATH = os.path.join(MODEL_PATH, "tokenizer")
PRETRAINED_MODEL = os.environ.get('PRETRAINED_MODEL')

TRAIN_PATH = os.path.join(DATA_PATH, "train_inputs_pos.pt")
DEV_PATH = os.path.join(DATA_PATH, "dev_inputs_pos.pt")
TEST_PATH = os.path.join(DATA_PATH, "test_inputs_pos.pt")

POS_TO_ID = {'': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'AUX': 4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 
             'PART': 10, 'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16, 'X': 17, '_': 18 }
ID_TO_POS = {v: k for k, v in POS_TO_ID.items()}

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
        # if idx == 0:  # Only print for the first sample
        #     # print(f"Input IDs (sample {idx}): {data['input_ids']}")
        #     # print(f"Attention Mask (sample {idx}): {data['attention_mask']}")
        #     # print(f"Labels (sample {idx}): {data['labels']}")
        #     # print(f"Max Label: {data['labels'].max()}, Min Label: {data['labels'].min()}")
        # return data

train_dataset = POSTaggingDataset(TRAIN_PATH)
dev_dataset = POSTaggingDataset(DEV_PATH)
test_dataset = POSTaggingDataset(TEST_PATH)



class BertWithCRF(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertWithCRF, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)  # BERT backbone
        self.dropout = torch.nn.Dropout(0.1)  # Dropout for regularization
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)  # Emission scores
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

            loss = -self.crf(logits,tags=labels, mask=attention_mask.byte())
            return loss
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return predictions



# Initialize model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH)
model = BertWithCRF(PRETRAINED_MODEL, num_labels=(len(POS_TO_ID)))
model.bert.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepSpeed Configuration
ds_config = {
    "train_micro_batch_size_per_gpu": 16,
    "gradient_accumulation_steps": 2,
    "fp16": {"enabled": True},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "zero_optimization": {
        "stage": 2
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)
data_collator = DataCollatorForTokenClassification(AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH), padding=True)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Training Loop with DeepSpeed
num_epochs = 20
for epoch in range(num_epochs):
    model_engine.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        loss = model_engine(input_ids, attention_mask, labels)
        print(loss)
        # Backward pass
        model_engine.backward(loss)
        model_engine.step()  # Update weights
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader)}")

# Validation Loop
model_engine.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for step, batch in enumerate(tqdm(dev_loader, desc="Validating")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model_engine(input_ids, attention_mask)  # Decode CRF predictions
        predictions.extend(preds)
        true_labels.extend(labels.cpu().tolist())

# Save Model
model_engine.save_checkpoint(POS_MODEL_PATH, client_state={"epoch": num_epochs})
print(f"Model saved to {POS_MODEL_PATH}.")
