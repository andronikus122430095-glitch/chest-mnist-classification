# train_resnet_fast.py
# Training ResNet-18 OPTIMIZED untuk KECEPATAN MAKSIMAL

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model_resnet import ResNet18
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import time

# --- Hyperparameter OPTIMIZED ---
EPOCHS = 20
BATCH_SIZE = 64  # ⚡ Lebih besar = lebih cepat
LEARNING_RATE = 0.001
PRETRAINED = True
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 4  # ⚡ Parallel data loading

print("="*70)
print(" TRAINING RESNET-18 OPTIMIZED - FAST MODE")
print("="*70)
print("🚀 Optimasi:")
print("  ✓ Batch Size 64 (4x lebih cepat)")
print("  ✓ Mixed Precision Training (2x lebih cepat)")
print("  ✓ Parallel Data Loading (num_workers=4)")
print("  ✓ Pin Memory untuk GPU transfer cepat")
print("="*70 + "\n")

def get_fast_data_loaders(batch_size):
    """DataLoader dengan optimasi kecepatan"""
    import torchvision.transforms as transforms
    from datareader import FilteredBinaryDataset
    from torch.utils.data import DataLoader
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    
    train_dataset = FilteredBinaryDataset('train', data_transform)
    val_dataset = FilteredBinaryDataset('test', data_transform)
    
    # ⚡ OPTIMASI: pin_memory + num_workers
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    return train_loader, val_loader, 2, 1

def train():
    # 1. Load Data dengan optimasi
    print("Loading dataset dengan optimasi kecepatan...")
    train_loader, val_loader, num_classes, in_channels = get_fast_data_loaders(BATCH_SIZE)
    
    # 2. Model
    print("\n--- Inisialisasi Model ResNet-18 ---")
    model = ResNet18(in_channels=in_channels, num_classes=num_classes, pretrained=PRETRAINED)
    
    # 3. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = model.to(device)
    
    # 4. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE*10,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader)
    )
    
    # ⚡ OPTIMASI: Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    use_amp = torch.cuda.is_available()
    
    if use_amp:
        print("⚡ Mixed Precision Training: ENABLED")
    
    # History
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n" + "="*70)
    print(" MEMULAI TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    total_time = 0
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            
            # ⚡ Mixed Precision Forward Pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.float().to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        val_loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    val_loss = criterion(outputs, labels)
                
                val_running_loss += val_loss.item()
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Timing
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        
        # Save best model
        save_marker = ""
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_resnet18_fast.pth')
            save_marker = "⚡ BEST!"
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) | "
              f"Train: {avg_train_loss:.4f}/{train_accuracy:.2f}% | "
              f"Val: {avg_val_loss:.4f}/{val_accuracy:.2f}% {save_marker}")
    
    avg_epoch_time = total_time / EPOCHS
    print("\n" + "="*70)
    print(" TRAINING SELESAI")
    print("="*70)
    print(f"⚡ Total Time: {total_time:.1f}s")
    print(f"⚡ Average Time/Epoch: {avg_epoch_time:.1f}s")
    print(f"✓ Best Val Accuracy: {best_val_acc:.2f}%")
    print("="*70 + "\n")
    
    # Plot
    plot_training_history(
        train_losses_history, val_losses_history,
        train_accs_history, val_accs_history,
        save_name='training_history_resnet18_fast.png'
    )
    
    visualize_random_val_predictions(
        model, val_loader, num_classes, count=10,
        save_name='val_predictions_resnet18_fast.png'
    )

if __name__ == '__main__':
    train()
