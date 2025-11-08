"""
Script untuk mengecek instalasi packages yang diperlukan
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\n" + "="*60)
print("Checking installed packages...")
print("="*60 + "\n")

packages = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'medmnist': 'MedMNIST'
}

all_installed = True

for package, name in packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name:<15} - version {version}")
    except ImportError as e:
        print(f"✗ {name:<15} - NOT INSTALLED")
        all_installed = False

print("\n" + "="*60)
if all_installed:
    print("✓ All packages are installed successfully!")
    print("\nYou can now run:")
    print("  python train_densenet.py")
else:
    print("✗ Some packages are missing. Please install them first.")
    print("\nRun this command:")
    print("  C:/Users/Asus/miniconda3/envs/andro/python.exe -m pip install medmnist matplotlib numpy")
print("="*60)
