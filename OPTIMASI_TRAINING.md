# 🚀 CARA MEMPERCEPAT TRAINING

## ⚠️ Masalah: Mengapa Training Lambat?

### 1. **Upsampling Berlebihan (PALING BERDAMPAK)**
- Gambar asli: 28x28 pixels
- Di-upsample ke: 224x224 pixels
- **Pembesaran data: 64x lipat!**
- Operasi ini dilakukan SETIAP forward pass

### 2. **Model Terlalu Besar**
- ResNet-18: ~11.2M parameter
- DenseNet-121: ~7.0M parameter
- Dataset filtered: hanya ~1000-2000 gambar
- **Overkill!** Model terlalu kompleks untuk dataset kecil

### 3. **Batch Size Terlalu Kecil**
- Batch size 16 → banyak iterasi per epoch
- GPU tidak optimal (underutilized)

### 4. **Tidak Ada GPU Optimization**
- Tidak pakai Mixed Precision Training
- Tidak pakai pin_memory
- Tidak pakai num_workers untuk parallel loading
- Tidak pakai torch.compile

### 5. **Data Augmentation Real-time**
- 4 jenis augmentasi dilakukan setiap epoch
- Memperlambat data loading

---

## ✅ SOLUSI: Cara Mempercepat Training

### 🎯 **Solusi #1: Gunakan File `train_resnet_fast.py`** (RECOMMENDED)

File ini sudah include semua optimasi:

```bash
python train_resnet_fast.py
```

**Optimasi yang diterapkan:**
- ✅ Batch size 64 (4x lebih cepat)
- ✅ Mixed Precision Training (2x lebih cepat di GPU)
- ✅ Parallel Data Loading (num_workers=4)
- ✅ Pin Memory untuk GPU transfer cepat
- ✅ torch.compile untuk PyTorch 2.0+
- ✅ OneCycleLR scheduler (konvergensi lebih cepat)

**Perkiraan speedup: 5-10x lebih cepat!**

---

### 🎯 **Solusi #2: Modifikasi Manual**

#### A. Perbesar Batch Size
```python
BATCH_SIZE = 64  # atau 128 jika GPU cukup besar
```

#### B. Tambahkan DataLoader Optimization
```python
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,        # ⚡ Parallel loading
    pin_memory=True,      # ⚡ Fast GPU transfer
    persistent_workers=True
)
```

#### C. Gunakan Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Di training loop:
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### D. Optimalkan Zero Grad
```python
optimizer.zero_grad(set_to_none=True)  # Lebih cepat dari zero_grad()
```

#### E. Gunakan AdamW + OneCycleLR
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.01,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader)
)
```

---

### 🎯 **Solusi #3: Gunakan Model Lebih Kecil**

Untuk dataset kecil (~1000-2000 gambar), model besar tidak perlu:

#### Opsi A: ResNet-18 Tanpa Upsampling
Modifikasi `model_resnet.py`:
```python
def forward(self, x):
    # HAPUS upsampling ini:
    # if x.shape[-1] != 224:
    #     x = nn.functional.interpolate(...)
    
    # Gunakan adaptive pooling untuk handle ukuran kecil
    x = self.resnet.conv1(x)
    # ... dst
```

#### Opsi B: Gunakan SimpleCNN
Model kecil (~100K parameter) sudah cukup untuk ChestMNIST:
```python
# Lihat model.py untuk SimpleCNN
```

---

## 📊 Perbandingan Kecepatan

| Konfigurasi | Waktu/Epoch | Speedup |
|-------------|-------------|---------|
| **Original (batch=16, no opt)** | ~60-120s | 1x |
| **+ Batch size 64** | ~20-40s | 3x |
| **+ Mixed Precision** | ~10-20s | 6x |
| **+ All optimizations** | ~5-10s | **10x** |

*Waktu tergantung pada GPU Anda

---

## 🔍 Cek Bottleneck Training Anda

Jalankan ini untuk analisis:

```python
import time
import torch

# Test forward pass speed
model.eval()
images = torch.randn(64, 1, 28, 28).cuda()

# Warmup
for _ in range(10):
    _ = model(images)

# Benchmark
start = time.time()
for _ in range(100):
    _ = model(images)
torch.cuda.synchronize()
end = time.time()

print(f"Time per batch: {(end-start)/100*1000:.2f}ms")
```

---

## 🚀 Quick Start: Mulai Training Cepat

```bash
# 1. Gunakan file optimized
python train_resnet_fast.py

# 2. Atau dengan parameter custom
python train_resnet_fast.py --batch-size 128 --epochs 20
```

---

## ⚙️ Tips Tambahan

### 1. Monitor GPU Usage
```bash
# Windows (jika punya NVIDIA GPU)
nvidia-smi -l 1
```

### 2. Reduce Augmentation
Untuk speed, disable augmentation:
```python
# Gunakan datareader.py (no aug) instead of datareader_augmented.py
```

### 3. Early Stopping
Sudah ada di code Anda, pastikan aktif:
```python
patience = 10  # Stop jika tidak improve
```

### 4. Reduce Epochs
Untuk testing cepat:
```python
EPOCHS = 10  # Instead of 30-40
```

---

## 📈 Expected Results

**Sebelum optimasi:**
- Training time: ~30-60 menit untuk 30 epochs
- Time/epoch: ~1-2 menit

**Setelah optimasi:**
- Training time: ~5-10 menit untuk 30 epochs  
- Time/epoch: ~10-20 detik

**Speedup total: 5-10x lebih cepat! 🚀**

---

## ❓ FAQ

**Q: GPU saya hanya punya 4GB VRAM, batch size 64 error?**
A: Turunkan ke 32 atau 48. Tetap lebih cepat dari 16.

**Q: Tidak punya GPU?**
A: Gunakan SimpleCNN (model.py) + batch size 32. ResNet terlalu berat untuk CPU.

**Q: Mixed precision error di CPU?**
A: Normal, AMP hanya untuk GPU. Code sudah handle ini otomatis.

**Q: Training masih lambat setelah optimasi?**
A: Kemungkinan:
   1. Masih pakai upsampling 224x224 → Hapus
   2. Model terlalu besar → Gunakan SimpleCNN
   3. Disk lambat → Pindah dataset ke SSD

---

## 📝 Checklist Optimasi

- [ ] Ganti ke `train_resnet_fast.py`
- [ ] Batch size minimal 64
- [ ] Mixed Precision enabled (GPU)
- [ ] num_workers=4
- [ ] pin_memory=True
- [ ] Hapus upsampling jika tidak perlu
- [ ] Gunakan AdamW + OneCycleLR
- [ ] Monitor dengan nvidia-smi

---

**Selamat training! 🚀**
