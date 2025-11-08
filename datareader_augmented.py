import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from medmnist import ChestMNIST

# --- Konfigurasi Kelas Biner ---
CLASS_A_IDX = 1  # 'Cardiomegaly'
CLASS_B_IDX = 7 # 'Pneumothorax'

NEW_CLASS_NAMES = {0: 'Cardiomegaly', 1: 'Pneumothorax'}
ALL_CLASS_NAMES = [
    'Atelectasis',        # 0
    'Cardiomegaly',       # 1
    'Effusion',           # 2
    'Infiltration',       # 3
    'Mass',               # 4
    'Nodule',             # 5
    'Pneumonia',          # 6
    'Pneumothorax',       # 7
    'Consolidation',      # 8
    'Edema',              # 9
    'Emphysema',          # 10
    'Fibrosis',           # 11
    'Pleural_Thickening', # 12
    'Hernia',             # 13
]

class FilteredBinaryDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        
        # Muat dataset lengkap
        full_dataset = ChestMNIST(split=split, transform=None, download=True)
        original_labels = full_dataset.labels

        # Cari indeks untuk gambar yang HANYA memiliki satu label yang kita inginkan
        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        # Simpan gambar dan label yang sudah dipetakan ulang
        self.images = []
        self.labels = []

        # Tambahkan data untuk kelas Cardiomegaly (dipetakan ke label 0)
        for idx in indices_a:
            self.images.append(full_dataset[idx][0])
            self.labels.append(0)

        # Tambahkan data untuk kelas Pneumothorax (dipetakan ke label 1)
        for idx in indices_b:
            self.images.append(full_dataset[idx][0])
            self.labels.append(1)
        
        print(f"Split: {split}")
        print(f"Jumlah Cardiomegaly (label 0): {len(indices_a)}")
        print(f"Jumlah Pneumothorax (label 1): {len(indices_b)}")
        print()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([label])


def get_data_loaders_augmented(batch_size):
    """
    Membuat data loaders dengan augmentasi untuk training set.
    Augmentasi membantu model lebih robust dan mengurangi overfitting.
    """
    
    # Augmentasi data untuk training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),           # Flip horizontal dengan probabilitas 50%
        transforms.RandomRotation(degrees=10),             # Rotasi random ±10 derajat
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translasi ±10%
        transforms.ColorJitter(brightness=0.2, contrast=0.2),       # Variasi brightness & contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    
    # Transform untuk validasi (tanpa augmentasi)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    train_dataset = FilteredBinaryDataset('train', train_transform)
    val_dataset = FilteredBinaryDataset('test', val_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    n_classes = 2
    n_channels = 1
    
    print("Dataset ChestMNIST berhasil difilter untuk klasifikasi biner (dengan Augmentasi)!")
    print(f"Kelas yang digunakan: {NEW_CLASS_NAMES[0]} (Label 0) dan {NEW_CLASS_NAMES[1]} (Label 1)")
    print(f"Jumlah data training: {len(train_dataset)}")
    print(f"Jumlah data validasi: {len(val_dataset)}")
    print("\nAugmentasi Training:")
    print("  - RandomHorizontalFlip (p=0.5)")
    print("  - RandomRotation (±10°)")
    print("  - RandomAffine (translate ±10%)")
    print("  - ColorJitter (brightness & contrast ±20%)")
    print()
    
    return train_loader, val_loader, n_classes, n_channels


def show_augmentation_examples(dataset, num_examples=5):
    """
    Menampilkan contoh gambar sebelum dan sesudah augmentasi.
    """
    # Transform tanpa augmentasi
    simple_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Transform dengan augmentasi
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    fig.suptitle("Perbandingan: Original (atas) vs Augmented (bawah)", fontsize=16)
    
    for i in range(num_examples):
        # Ambil gambar random
        idx = np.random.randint(0, len(dataset))
        original_img = dataset.images[idx]
        
        # Original
        img_original = simple_transform(original_img)
        axes[0, i].imshow(img_original.squeeze(), cmap='gray')
        axes[0, i].set_title(f"Original #{i+1}")
        axes[0, i].axis('off')
        
        # Augmented
        img_augmented = aug_transform(original_img)
        axes[1, i].imshow(img_augmented.squeeze(), cmap='gray')
        axes[1, i].set_title(f"Augmented #{i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
    print("Contoh augmentasi disimpan sebagai 'augmentation_examples.png'")
    plt.show()


if __name__ == '__main__':
    print("Memuat dataset dengan augmentasi...")
    
    # Test data loader
    train_loader, val_loader, num_classes, in_channels = get_data_loaders_augmented(batch_size=16)
    
    # Tampilkan contoh augmentasi
    print("\n--- Menampilkan Contoh Augmentasi ---")
    train_dataset = FilteredBinaryDataset('train', transform=None)
    if len(train_dataset) > 0:
        show_augmentation_examples(train_dataset, num_examples=5)
    else:
        print("Dataset tidak berisi sampel untuk kelas yang dipilih.")
