# train_cpu_optimized.py
# OPTIMIZED UNTUK CPU - Training cepat tanpa GPU

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time

from datareader import FilteredBinaryDataset

# --- SIMPLIFIED MODEL untuk CPU ---
class SimpleCNN(nn.Module):
    """Model RINGAN - ~100K parameters (100x lebih kecil dari ResNet!)"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers - langsung di 28x28, TANPA upsampling!
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected
        # Setelah 3 pooling: 28 -> 14 -> 7 -> 3 (rounded down)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # TIDAK ADA UPSAMPLING - langsung proses 28x28!
        x = self.pool(self.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(self.relu(self.conv2(x)))  # 14 -> 7
        x = self.pool(self.relu(self.conv3(x)))  # 7 -> 3
        
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# --- CONFIG UNTUK CPU ---
EPOCHS = 15  # Lebih sedikit untuk cepat
BATCH_SIZE = 128  # Lebih besar OK di CPU karena model kecil
LEARNING_RATE = 0.001

print("="*60)
print(" TRAINING SIMPLE CNN - CPU OPTIMIZED")
print("="*60)
print("⚡ Optimasi untuk CPU:")
print("  ✓ Model ringan (~100K params vs 11M ResNet)")
print("  ✓ TANPA upsampling (28x28 langsung)")
print("  ✓ Batch size besar (128)")
print("  ✓ Epochs minimal (15)")
print(f"\nEstimasi waktu: 2-5 menit total")
print("="*60 + "\n")

def train():
    # 1. Data
    print("Loading data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    
    train_dataset = FilteredBinaryDataset('train', transform)
    val_dataset = FilteredBinaryDataset('test', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Batches/epoch: {len(train_loader)}\n")
    
    # 2. Model RINGAN
    print("Initializing SimpleCNN...")
    model = SimpleCNN()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (vs 11,000,000 ResNet!)")
    print("Model ready!\n")
    
    # 3. Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training
    print("="*60)
    print(" STARTING TRAINING")
    print("="*60)
    
    best_val_acc = 0.0
    total_start = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            labels = labels.float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1} - Batch {batch_idx+1}/{len(train_loader)}", end='\r')
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.float()
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Save best
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_simple_cnn_cpu.pth')
            marker = "⭐ BEST!"
        
        print(f"[{epoch+1:2d}/{EPOCHS}] {epoch_time:4.1f}s | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {train_acc:5.2f}%/{val_acc:5.2f}% {marker}")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print("="*60)
    print(f"⚡ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"⚡ Avg time/epoch: {total_time/EPOCHS:.1f}s")
    print(f"✓ Best accuracy: {best_val_acc:.2f}%")
    print(f"✓ Model saved: best_simple_cnn_cpu.pth")
    print("="*60)
    
    # Quick test
    print("\n--- Validation Breakdown ---")
    model.eval()
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predicted = (outputs > 0).float()
            
            for i in range(len(labels)):
                label = int(labels[i].item())
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    print(f"Cardiomegaly: {100*class_correct[0]/class_total[0]:.1f}% ({class_correct[0]}/{class_total[0]})")
    print(f"Pneumothorax: {100*class_correct[1]/class_total[1]:.1f}% ({class_correct[1]}/{class_total[1]})")
    print("="*60)

if __name__ == '__main__':
    train()
