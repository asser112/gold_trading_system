#!/usr/bin/env python3
"""
Transformer Model Training
- Uses sequences of 60 M5 candles
- Architecture: 4 TransformerEncoder layers, 8 heads, FF=256, dropout=0.2
- Output: softmax over {short, hold, long} (0,1,2)
- Training with AdamW, early stopping
- Saves model and attention maps (optional)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yaml
import logging
import joblib

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.2, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.transformer(x)          # (batch, seq_len, d_model)
        x = x[:, -1, :]                  # use last time step
        x = self.fc(x)
        return self.softmax(x)

class SequenceDataset(Dataset):
    def __init__(self, features, targets, seq_len=60):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len-1]  # target for last candle
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def main():
    # Load features and targets
    df = pd.read_parquet('data/processed/features_target_m5.parquet')
    df = df.dropna()
    feature_cols = [c for c in df.columns if c != 'target']
    X = df[feature_cols].values
    y = df['target'].values + 1  # convert -1,0,1 to 0,1,2

    # Chronological split (80% train, 20% test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    seq_len = config['models']['transformer']['seq_len']
    train_dataset = SequenceDataset(X_train, y_train, seq_len)
    test_dataset = SequenceDataset(X_test, y_test, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config['models']['transformer']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['models']['transformer']['batch_size'], shuffle=False)

    input_dim = X.shape[1]
    model = TransformerModel(
        input_dim=input_dim,
        d_model=config['models']['transformer']['d_model'],
        nhead=config['models']['transformer']['nhead'],
        num_layers=config['models']['transformer']['num_layers'],
        dim_feedforward=config['models']['transformer']['dim_feedforward'],
        dropout=config['models']['transformer']['dropout']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['models']['transformer']['lr'])
    epochs = config['models']['transformer']['epochs']
    patience = config['models']['transformer']['patience']
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        val_loss /= len(test_loader)
        accuracy = 100 * correct / total
        logger.info(f'Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}, Accuracy={accuracy:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/transformer/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Save final model
    torch.save(model.state_dict(), 'models/transformer/final_model.pth')
    logger.info("Transformer training completed.")

if __name__ == '__main__':
    main()