import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import joblib
from torchmetrics import Accuracy, F1Score
from model import SentimentClassifier

# Load dataset
data = pd.read_csv("sentiment_analysis.csv")
texts = data.iloc[:, 4].astype(str).tolist()
labels = data.iloc[:, 5].astype(str).tolist()

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
joblib.dump(label_encoder, "label_encoder.pkl")

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Dataloaders
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
num_labels = len(label_encoder.classes_)
model = SentimentClassifier(num_labels=num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Initialize metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Determine task type for F1Score
task_type = "multiclass" if num_labels > 2 else "binary"
accuracy_metric = Accuracy(task=task_type, num_classes=num_labels).to(device)
f1_metric = F1Score(task=task_type, num_classes=num_labels, average='weighted').to(device)

def evaluate_model(model, data_loader, criterion, accuracy_metric, f1_metric):
    # """Evaluate the model and return loss, accuracy, and F1 score"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu())
            all_labels.extend(labels.cpu())
    
    # Convert to tensors for torchmetrics
    all_predictions = torch.tensor(all_predictions).to(device)
    all_labels = torch.tensor(all_labels).to(device)
    
    # Calculate metrics
    accuracy = accuracy_metric(all_predictions, all_labels)
    f1 = f1_metric(all_predictions, all_labels)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy.item(), f1.item()

# Training loop with evaluation
print("Starting training...")
Epochs = 10
for epoch in range(Epochs):  # small for demo
    model.train()
    train_loss = 0
    train_predictions = []
    train_labels_list = []
    
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Collect predictions and labels for training metrics
        predictions = torch.argmax(outputs, dim=1)
        train_predictions.extend(predictions.cpu())
        train_labels_list.extend(labels.cpu())
    
    # Calculate training metrics
    train_predictions_tensor = torch.tensor(train_predictions)
    train_labels_tensor = torch.tensor(train_labels_list)
    train_accuracy = accuracy_metric(train_predictions_tensor, train_labels_tensor).item()
    train_f1 = f1_metric(train_predictions_tensor, train_labels_tensor).item()
    avg_train_loss = train_loss / len(train_loader)
    
    # Evaluate on validation set
    val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, accuracy_metric, f1_metric)
    
    print(f"Epoch {epoch+1}/{Epochs}:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    print("-" * 60)

# Save model
torch.save(model.state_dict(), "bert_sentiment.pt")
print("Model saved successfully!")

# Final evaluation
print("\nFinal Validation Results:")
val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, accuracy_metric, f1_metric)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")