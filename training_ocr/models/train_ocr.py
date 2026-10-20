#!/usr/bin/env python3
"""
Thai License Plate CRNN (CNN + BiLSTM + CTC) OCR Model Training Script.
Trains an end-to-end character recognition model on synthetic and real plate sequences.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .char_vocab import get_vocab_size, decode_indices
from .dataset_loader import ThaiPlateDataset


class CRNN(nn.Module):
    """
    Standard CRNN Architecture:
    - 7-Layer VGG-style CNN feature extractor with MaxPool
    - 2-Layer Bidirectional LSTM
    - Linear projection to vocabulary size
    """

    def __init__(self, num_classes: int, in_channels: int = 1, hidden_size: int = 256):
        super(CRNN, self).__init__()
        
        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/2, W/2)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (H/4, W/4)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (H/8, W/4)
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (H/16, W/4)
            
            nn.Conv2d(512, 512, kernel_size=2, padding=0),    # (1, W/4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bidirectional LSTM Sequence Model
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=False),
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Channel, Height, Width]
        features = self.cnn(x)
        b, c, h, w = features.size()
        assert h == 1, f"Height after CNN must be 1, got {h}"
        
        # Squeeze height and permute for RNN: [Time, Batch, Channels]
        features = features.squeeze(2).permute(2, 0, 1)
        
        # RNN
        recurrent, _ = self.rnn[0](features)
        
        # Linear projection
        logits = self.fc(recurrent)  # [Time, Batch, NumClasses]
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets, input_lens, target_lens) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        input_lens = torch.tensor(input_lens, dtype=torch.long)
        target_lens = torch.tensor(target_lens, dtype=torch.long)

        optimizer.zero_grad()
        
        logits = model(images)  # [T, B, C]
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lens, target_lens)
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train Thai License Plate CRNN OCR Model")
    parser.add_argument("--data-dir", type=str, default="../synthetic_data/augmented_plates", help="Path to training plate images")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--img-h", type=int, default=32, help="Input height")
    parser.add_argument("--img-w", type=int, default=128, help="Input width")
    parser.add_argument("--output", type=str, default="thai_plate_crnn.pt", help="Checkpoint output name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("       THAI LICENSE PLATE OCR (CRNN + CTC) TRAINING")
    print("=" * 65)
    print(f"Device        : {device}")
    print(f"Dataset Dir   : {args.data_dir}")
    print(f"Vocabulary    : {get_vocab_size()} classes")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch}")
    print("=" * 65)

    dataset = ThaiPlateDataset(args.data_dir, img_h=args.img_h, img_w=args.img_w)
    if len(dataset) == 0:
        print(f"[WARN] No samples found in {args.data_dir}.")
        print("Please generate synthetic plates first via: python ../synthetic_data/augment_license_plate.py")
        sys.exit(0)

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    model = CRNN(num_classes=get_vocab_size()).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] - CTC Loss: {loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"\n[SUCCESS] Saved trained CRNN OCR weights to: {args.output}")


if __name__ == "__main__":
    main()
