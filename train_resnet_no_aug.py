# train_resnet_no_aug.py
# Training ResNet-18 TANPA AUGMENTASI - untuk akurasi maksimal

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES  # Tanpa augmentasi!
from model_resnet import ResNet18
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import time

# --- Hyperparameter ---
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
PRETRAINED = True
WEIGHT_DECAY = 1e-3

print("="*70)
print(" TRAINING RESNET-18 TANPA AUGMENTASI")
print("="*70)
print("Mode: Fokus pada akurasi maksimal tanpa data augmentation")
print(f"  - Learning Rate: {LEARNING_RATE}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Weight Decay: {WEIGHT_DECAY}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Pretrained: {PRETRAINED}")
print("="*70 + "\n")

def train():
    # 1. Memuat Data TANPA AUGMENTASI
    print("Loading dataset TANPA AUGMENTASI...")
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model ResNet-18
    print("\n--- Inisialisasi Model ResNet-18 ---")
    model = ResNet18(in_channels=in_channels, num_classes=num_classes, pretrained=PRETRAINED)
    print(f"ResNet-18 (Pretrained: {PRETRAINED})")
    
    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 3. Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nMenggunakan device: {device}")
    model = model.to(device)
    
    # 4. Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # Gunakan SGD dengan momentum untuk stabilitas
    optimizer = optim.SGD(model.parameters(), 
                         lr=LEARNING_RATE, 
                         momentum=0.9,
                         weight_decay=WEIGHT_DECAY,
                         nesterov=True)  # Nesterov untuk konvergensi lebih baik
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Early stopping
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n" + "="*70)
    print(" MEMULAI TRAINING")
    print("="*70)
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Early stopping patience: {patience} epochs")
    print("="*70 + "\n")
    
    # 5. Training Loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # --- Fase Training ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress indicator
            if (batch_idx + 1) % 20 == 0:
                current_acc = 100 * train_correct / train_total
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f} - Acc: {current_acc:.2f}%", end='\r')
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # Hitung waktu epoch
        epoch_time = time.time() - epoch_start_time
        
        # Simpan model terbaik
        save_marker = ""
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_resnet18_no_aug.pth')
            save_marker = "✓ BEST ACC!"
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_resnet18_no_aug_loss.pth')
        else:
            patience_counter += 1
        
        # Display learning rate
        lr_info = f" | LR: {new_lr:.6f}"
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s){lr_info} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% {save_marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠ Early stopping! No improvement for {patience} epochs.")
            break

    print("\n" + "="*70)
    print(" TRAINING SELESAI")
    print("="*70)
    print(f"Akurasi Validasi Terbaik: {best_val_acc:.2f}%")
    print(f"Loss Validasi Terbaik: {best_val_loss:.4f}")
    print("="*70 + "\n")
    
    # Simpan model terakhir
    torch.save(model.state_dict(), 'last_resnet18_no_aug.pth')
    
    # Tampilkan plot
    print("\n--- Membuat Plot Training History ---")
    plot_training_history(
        train_losses_history, val_losses_history, 
        train_accs_history, val_accs_history, 
        save_name='training_history_resnet18_no_aug.png'
    )

    # Visualisasi prediksi
    print("\n--- Membuat Visualisasi Prediksi ---")
    visualize_random_val_predictions(
        model, val_loader, num_classes, count=10, 
        save_name='val_predictions_resnet18_no_aug.png'
    )
    
    print("\n" + "="*70)
    print(" FILE YANG DIHASILKAN")
    print("="*70)
    print("  1. best_resnet18_no_aug.pth - Model akurasi terbaik")
    print("  2. best_resnet18_no_aug_loss.pth - Model loss terbaik")
    print("  3. last_resnet18_no_aug.pth - Model epoch terakhir")
    print("  4. training_history_resnet18_no_aug.png - Grafik")
    print("  5. val_predictions_resnet18_no_aug.png - Visualisasi")
    print("="*70 + "\n")


if __name__ == '__main__':
    train()
