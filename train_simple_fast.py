# train_simple_fast.py
# Training SIMPLE & FAST - Minimal dependencies, maksimal kecepatan!

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Import lokal
from datareader import FilteredBinaryDataset
from model_resnet import ResNet18

# --- Config ---
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Set 0 untuk avoid multiprocessing issues

print("="*60)
print(" TRAINING RESNET-18 - SIMPLE FAST MODE")
print("="*60)
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print("="*60 + "\n")

def train():
    # 1. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("WARNING: Running on CPU - will be slower!\n")
    
    # 2. Data
    print("Loading data...")
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    
    train_dataset = FilteredBinaryDataset('train', transform)
    val_dataset = FilteredBinaryDataset('test', transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    # 3. Model
    print("Initializing model...")
    model = ResNet18(in_channels=1, num_classes=2, pretrained=True)
    model = model.to(device)
    print("Model loaded!\n")
    
    # 4. Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Mixed precision
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("⚡ Mixed Precision: ENABLED\n")
    
    # 5. Training loop
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
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        
        # Save best
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet18_simple_fast.pth')
            marker = "⭐ BEST!"
        
        print(f"[{epoch+1}/{EPOCHS}] {epoch_time:.1f}s | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {train_acc:.2f}%/{val_acc:.2f}% {marker}")
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg time/epoch: {total_time/EPOCHS:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: best_resnet18_simple_fast.pth")
    print("="*60)

if __name__ == '__main__':
    train()
