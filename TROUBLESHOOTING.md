# 🚨 TROUBLESHOOTING: Training Terlihat Stuck

## Masalah Anda

Training **SEBENARNYA BERJALAN**, tapi **sangat lambat** karena:

### ❌ Penyebab:
1. **Tidak ada GPU** - training di CPU
2. **Model terlalu besar** - ResNet-18 dengan 11M parameter
3. **Upsampling** - setiap batch, gambar diperbesar 64x (28x28 → 224x224)

### ⏱️ Estimasi Waktu REAL di CPU:

| Model | Time/Epoch | Total (20 epochs) |
|-------|------------|-------------------|
| **ResNet-18** | 5-15 menit | **1.5 - 5 JAMKK!** 😱 |
| **SimpleCNN** | 10-30 detik | **5-10 menit** ✅ |

---

## ✅ SOLUSI: Gunakan SimpleCNN

### Cara 1: Batch File (TERMUDAH)
```bash
RUN_TRAINING.bat
```
Lalu pilih opsi 1 (SimpleCNN)

### Cara 2: Manual
```bash
python train_cpu_optimized.py
```

**PENTING**: Setelah run, **TUNGGU 1-2 MENIT** untuk loading, lalu training akan dimulai!

---

## 📊 Progress Tracking

Anda akan melihat output seperti ini:

```
============================================================
 TRAINING SIMPLE CNN - CPU OPTIMIZED
============================================================
⚡ Optimasi untuk CPU:
  ✓ Model ringan (~100K params vs 11M ResNet)
  ✓ TANPA upsampling (28x28 langsung)
  ✓ Batch size besar (128)
  ✓ Epochs minimal (15)

Estimasi waktu: 2-5 menit total
============================================================

Loading data...
Split: train
Jumlah Cardiomegaly (label 0): 754
Jumlah Pneumothorax (label 1): 1552

Split: test
Jumlah Cardiomegaly (label 0): 243
Jumlah Pneumothorax (label 1): 439

Batches/epoch: 19

Initializing SimpleCNN...
Parameters: 142,081 (vs 11,000,000 ResNet!)
Model ready!

============================================================
 STARTING TRAINING
============================================================
[ 1/15]  18.3s | Loss: 0.6432/0.5123 | Acc: 65.23%/72.45% ⭐ BEST!
[ 2/15]  17.8s | Loss: 0.4821/0.4567 | Acc: 73.12%/76.83% ⭐ BEST!
...
```

**Jika loading lebih dari 2 menit, itu artinya PyTorch initialization lambat. NORMAL!**

---

## 🔥 Opsi TERCEPAT: Google Colab (GRATIS GPU!)

Jika training tetap lambat, gunakan Google Colab:

### 1. Upload files ke Google Drive:
- `datareader.py`
- `model_resnet.py`  
- `train_simple_fast.py`
- `utils.py`

### 2. Buka Google Colab:
```
https://colab.research.google.com
```

### 3. Copy paste code ini di cell pertama:
```python
# Install dependencies
!pip install medmnist

# Upload files dari drive atau local
from google.colab import files
# files.upload() # uncomment jika upload manual

# Run training
!python train_simple_fast.py
```

### 4. Aktifkan GPU:
- Runtime → Change runtime type → GPU (T4)
- Gratis dan **100x lebih cepat**!

**Dengan GPU di Colab**: 5-10 detik per epoch = **2-3 menit total!** 🚀

---

## ⚙️ Quick Checklist

- [ ] Pakai `train_cpu_optimized.py` (SimpleCNN)
- [ ] Atau `RUN_TRAINING.bat` → pilih 1
- [ ] Tunggu 1-2 menit untuk initialization
- [ ] Jangan panic jika tidak ada output segera
- [ ] Jika masih lambat → Gunakan Google Colab dengan GPU

---

## 📞 Debugging Commands

Jika masih stuck, jalankan ini untuk diagnostic:

```bash
# Check Python & PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Test model forward pass speed
python -c "import torch; from train_cpu_optimized import SimpleCNN; import time; m=SimpleCNN(); x=torch.randn(32,1,28,28); s=time.time(); m(x); print(f'Time: {time.time()-s:.3f}s')"

# Check if process running
tasklist | findstr python
```

---

## 🎯 Expected Output Timeline

| Time | Event |
|------|-------|
| 0s | Command run |
| 0-30s | PyTorch loading (silent) |
| 30-60s | MedMNIST download & loading |
| 60-120s | Model initialization |
| 120s+ | **TRAINING STARTS** (you see epoch progress!) |

**TOTAL waiting before training: 1-2 menit**

Jika Anda Ctrl+C sebelum 2 menit → training belum sempat mulai!

---

## ✅ Konfirmasi Training Berjalan

Anda tahu training berjalan jika melihat:
```
[ 1/15]  18.3s | Loss: ... | Acc: ...
```

Jika tidak muncul setelah 3 menit → ada masalah.

---

**BOTTOM LINE**: Pilih **SimpleCNN** (`train_cpu_optimized.py`), tunggu 2 menit, dan training akan selesai dalam 5-10 menit total! 🚀
