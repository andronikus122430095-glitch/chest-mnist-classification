# model_resnet.py

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    """
    ResNet-18 model untuk klasifikasi gambar medis ChestMNIST.
    ResNet-18 lebih ringan dan cepat dibanding DenseNet, cocok untuk CPU.
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load ResNet-18 pre-trained
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Modifikasi layer pertama untuk menerima grayscale (1 channel)
        if in_channels == 1:
            # Ambil weight dari layer pertama yang sudah pre-trained (3 channels)
            original_conv = self.resnet.conv1
            # Buat conv layer baru dengan 1 channel input
            self.resnet.conv1 = nn.Conv2d(
                1, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Inisialisasi weight dengan rata-rata dari 3 channel RGB
            if pretrained:
                with torch.no_grad():
                    self.resnet.conv1.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
        
        # Modifikasi fully connected layer untuk binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1 if num_classes == 2 else num_classes)
        
        # Tambahkan dropout untuk regularisasi
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Upsample dari 28x28 ke 224x224 (ukuran standar ImageNet)
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Forward pass melalui ResNet
        # ResNet-18 sudah include adaptive avg pooling di dalamnya
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        
        return x


# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'ResNet-18' ---")
    
    resnet_model = ResNet18(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=False)
    print("Arsitektur Model ResNet-18 (ringkasan):")
    print(f"Input channels: {IN_CHANNELS}")
    print(f"Output classes: {NUM_CLASSES}")
    
    dummy_input = torch.randn(4, IN_CHANNELS, 28, 28)
    output = resnet_model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'ResNet-18' berhasil.")
    
    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in resnet_model.parameters())
    trainable_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("Perbandingan dengan model lain:")
    print("  - ResNet-18:    ~11.2M parameters")
    print("  - DenseNet-121: ~7.0M parameters")
    print("  - SimpleCNN:    ~0.12M parameters")
    print("="*60)
