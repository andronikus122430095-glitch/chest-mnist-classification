# Training DenseNet dengan Augmentasi Data

## 📁 File-File Baru

### 1. **model_densenet.py**
File yang berisi model DenseNet121 untuk klasifikasi ChestMNIST.

**Fitur:**
- Menggunakan DenseNet121 pre-trained dari ImageNet
- Modifikasi layer pertama untuk menerima grayscale (1 channel)
- Upsampling otomatis dari 28x28 ke 224x224
- Dropout 0.5 untuk regularisasi
- Binary classification output

**Cara test:**
```bash
python model_densenet.py
```

---

### 2. **datareader_augmented.py**
File yang berisi fungsi untuk load data dengan augmentasi.

**Augmentasi yang diterapkan pada training set:**
- `RandomHorizontalFlip` (probabilitas 50%)
- `RandomRotation` (±10 derajat)
- `RandomAffine` (translasi ±10%)
- `ColorJitter` (brightness & contrast ±20%)

**Validation set:** Tanpa augmentasi (hanya normalisasi)

**Cara test & visualisasi augmentasi:**
```bash
python datareader_augmented.py
```

---

### 3. **train_densenet.py**
Script training lengkap untuk DenseNet dengan augmentasi data.

**Hyperparameter:**
- Epochs: 50
- Batch Size: 16
- Learning Rate: 0.0001
- Weight Decay: 1e-4
- Pretrained: True

**Fitur:**
- GPU/CPU detection otomatis
- Learning rate scheduler (ReduceLROnPlateau)
- Save 3 model:
  - `best_densenet_model.pth` (akurasi terbaik)
  - `best_densenet_model_loss.pth` (loss terbaik)
  - `last_densenet_model.pth` (epoch terakhir)
- Visualisasi training history
- Visualisasi prediksi validation

---

## 🚀 Cara Menjalankan Training DenseNet

### 1. Install dependencies (jika belum):
```bash
pip install torch torchvision medmnist matplotlib numpy
```

### 2. Jalankan training:
```bash
python train_densenet.py
```

### 3. Monitor progress:
Training akan menampilkan progress per epoch:
```
Epoch [1/50] | Train Loss: 0.6234 | Train Acc: 65.23% | Val Loss: 0.5891 | Val Acc: 68.45% | ✓ BEST MODEL SAVED!
```

---

## 📊 Output yang Dihasilkan

Setelah training selesai, Anda akan mendapatkan:

1. **best_densenet_model.pth** - Model dengan validation accuracy terbaik
2. **best_densenet_model_loss.pth** - Model dengan validation loss terbaik
3. **last_densenet_model.pth** - Model pada epoch terakhir
4. **training_history_densenet.png** - Grafik loss & accuracy
5. **val_predictions_densenet.png** - Visualisasi 10 prediksi random

---

## 🔄 Perbandingan dengan Model Lama

| Aspek | SimpleCNN (model lama) | DenseNet (model baru) |
|-------|------------------------|----------------------|
| **File Training** | `train.py` | `train_densenet.py` |
| **File Model** | `model.py` | `model_densenet.py` |
| **File DataReader** | `datareader.py` | `datareader_augmented.py` |
| **Augmentasi** | ❌ Tidak ada | ✅ 4 jenis augmentasi |
| **Pre-trained** | ❌ Tidak | ✅ ImageNet weights |
| **Parameters** | ~120K | ~7M |
| **Batch Size** | 24 | 16 |
| **Epochs** | 40 | 50 |

---

## 💡 Tips

### Jika training terlalu lambat di CPU:
1. Kurangi batch size di `train_densenet.py`:
   ```python
   BATCH_SIZE = 8  # dari 16
   ```

2. Atau nonaktifkan pretrained weights:
   ```python
   PRETRAINED = False
   ```

### Jika ingin mengubah augmentasi:
Edit di file `datareader_augmented.py`, fungsi `get_data_loaders_augmented()`:
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    # Tambahkan augmentasi lain di sini
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5]),
])
```

---

## 🎯 Load Model untuk Inferensi

```python
import torch
from model_densenet import DenseNet

# Inisialisasi model
model = DenseNet(in_channels=1, num_classes=2, pretrained=False)

# Load weights terbaik
model.load_state_dict(torch.load('best_densenet_model.pth'))
model.eval()

# Gunakan untuk prediksi
with torch.no_grad():
    output = model(your_image_tensor)
    prediction = (output > 0).float()
```

---

## 📝 Catatan Penting

1. **File lama tetap utuh!** 
   - `train.py`, `model.py`, `datareader.py` tidak berubah
   - Anda masih bisa menjalankan SimpleCNN dengan `python train.py`

2. **GPU sangat direkomendasikan** untuk DenseNet karena modelnya lebih besar

3. **Pretrained weights** akan otomatis download saat pertama kali training

4. **Augmentasi hanya diterapkan pada training set**, validation tetap original untuk evaluasi yang fair

---

## ❓ Troubleshooting

**Q: Error "CUDA out of memory"**
A: Kurangi batch size atau gunakan CPU dengan setting di awal script:
```python
device = torch.device('cpu')
```

**Q: Training sangat lambat**
A: 
- Pastikan menggunakan GPU jika tersedia
- Kurangi batch size
- Atau set `PRETRAINED = False` untuk training from scratch

**Q: Ingin melihat contoh augmentasi**
A: Jalankan:
```bash
python datareader_augmented.py
```

---

## 📧 Support

Jika ada pertanyaan atau issue, silakan cek dokumentasi atau hubungi tim pengembang.

**Happy Training! 🎉**
