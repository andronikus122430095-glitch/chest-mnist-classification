# train_resnet.py
# Training script untuk ResNet-18 dengan Augmentasi Data

import torch
import torch.nn as nn
import torch.optim as optim
from datareader_augmented import get_data_loaders_augmented, NEW_CLASS_NAMES
from model_resnet import ResNet18
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import time

# --- Hyperparameter ---
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PRETRAINED = True    # Gunakan pretrained weights dari ImageNet
WEIGHT_DECAY = 1e-4  # L2 regularization

print("="*70)
print(" TRAINING RESNET-18 DENGAN AUGMENTASI DATA")
print("="*70)
print("Konfigurasi:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Pretrained: {PRETRAINED}")
print(f"  - Learning Rate: {LEARNING_RATE}")
print(f"  - Weight Decay: {WEIGHT_DECAY}")
print("="*70 + "\n")

def train():
    # 1. Memuat Data dengan Augmentasi
    train_loader, val_loader, num_classes, in_channels = get_data_loaders_augmented(BATCH_SIZE)
    
    # 2. Inisialisasi Model ResNet-18
    print("\n--- Inisialisasi Model ResNet-18 ---")
    model = ResNet18(in_channels=in_channels, num_classes=num_classes, pretrained=PRETRAINED)
    print(f"ResNet-18 (Pretrained: {PRETRAINED})")
    
    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 3. Deteksi device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nMenggunakan device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    model = model.to(device)
    
    # 4. Mendefinisikan Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n" + "="*70)
    print(" MEMULAI TRAINING")
    print("="*70)
    print(f"Total batches per epoch: {len(train_loader)}")
    print("="*70 + "\n")
    
    # 5. Training Loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
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
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Progress indicator setiap 10 batch
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}", end='\r')
        
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
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Hitung waktu epoch
        epoch_time = time.time() - epoch_start_time
        
        # Simpan model terbaik berdasarkan validation accuracy
        save_marker = ""
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_resnet18_model.pth')
            save_marker = "✓ BEST ACC!"
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_resnet18_model_loss.pth')
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% {save_marker}")

    print("\n" + "="*70)
    print(" TRAINING SELESAI")
    print("="*70)
    print(f"Akurasi Validasi Terbaik: {best_val_acc:.2f}%")
    print(f"Loss Validasi Terbaik: {best_val_loss:.4f}")
    print("="*70 + "\n")
    
    # Simpan model terakhir
    torch.save(model.state_dict(), 'last_resnet18_model.pth')
    print("Model terakhir disimpan sebagai 'last_resnet18_model.pth'")
    
    # Tampilkan plot
    print("\n--- Membuat Plot Training History ---")
    plot_training_history(
        train_losses_history, val_losses_history, 
        train_accs_history, val_accs_history, 
        save_name='training_history_resnet18.png'
    )

    # Visualisasi prediksi
    print("\n--- Membuat Visualisasi Prediksi ---")
    visualize_random_val_predictions(
        model, val_loader, num_classes, count=10, 
        save_name='val_predictions_resnet18.png'
    )
    
    print("\n" + "="*70)
    print(" SELESAI!")
    print("="*70)
    print("\nFile yang dihasilkan:")
    print("  1. best_resnet18_model.pth - Model dengan akurasi terbaik")
    print("  2. best_resnet18_model_loss.pth - Model dengan loss terbaik")
    print("  3. last_resnet18_model.pth - Model epoch terakhir")
    print("  4. training_history_resnet18.png - Grafik training history")
    print("  5. val_predictions_resnet18.png - Visualisasi prediksi")
    print("="*70 + "\n")


if __name__ == '__main__':
    train()
