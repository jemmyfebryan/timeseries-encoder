# Google Colab Training Guide

This guide shows how to train the Dual-Head Siamese Triplet TCN on Google Colab with free GPU.

## Quick Start

1. Upload your data to Google Drive
2. Open this notebook in Colab
3. Run all cells
4. Download trained model

## Setup

### Prerequisites

- Google account with Google Drive
- Your asset price CSV files in Google Drive
- Your `my_config.py` configuration file

### Step 1: Upload Data to Google Drive

1. Create folder structure in Google Drive:
   ```
   MyDrive/
   └── alpha_prices_csv/
       ├── alpha_prices_netuid_1.csv
       ├── alpha_prices_netuid_2.csv
       ...
       └── alpha_prices_netuid_128.csv
   ```

2. Upload your `my_config.py` to Google Drive (root)

### Step 2: Create Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click "File → New notebook"
3. Name it `train_siamese_tcn.ipynb`
4. Runtime → Change runtime type → **T4 GPU** (or other GPU)
5. Connect to GPU (upper right)

### Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
!ls /content/drive/MyDrive
```

### Step 4: Install Dependencies

```python
# Clone your repo (if public)
!git clone https://github.com/your-username/timeseries-encoder.git
%cd timeseries-encoder

# OR upload files manually
# Files → Upload → Upload notebook
```

```python
# Install dependencies
!pip install torch numpy pandas tensorboard tqdm -q

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Training

### Option 1: Train from Scratch

```python
# Set working directory
import os
os.chdir('/content/drive/MyDrive/timeseries-encoder')

# Update config to use Google Drive paths
# In my_config.py:
# data_dir = "./alpha_prices_csv"
# checkpoint_dir = "/content/drive/MyDrive/checkpoints"
# memory_bank_dir = "/content/drive/MyDrive/memory_bank"

# Train
!python main.py --config my_config.py --mode full
```

### Option 2: Resume from Existing Checkpoint

```python
# If you have checkpoints in Google Drive
!python main.py \
    --config my_config.py \
    --mode train \
    --checkpoint /content/drive/MyDrive/checkpoints/checkpoint_epoch_15.pt
```

### Option 3: Training Only (No Inference)

```python
!python main.py \
    --config my_config.py \
    --mode train \
    --checkpoint /content/drive/MyDrive/checkpoints/checkpoint_epoch_15.pt
```

### Option 4: Inference Only (Using Trained Model)

```python
!python main.py \
    --config my_config.py \
    --mode inference \
    --checkpoint /content/drive/MyDrive/checkpoints/best_model.pt \
    --memory-bank-path /content/drive/MyDrive/memory_bank/memory_bank.pkl
```

## Monitoring Training

### View Training Logs

```python
# Tail log file in real-time
!tail -f training.log
```

### Launch TensorBoard

```python
# Start TensorBoard in background
%load_ext tensorboard
%tensorboard --logdir runs --port 6006

# Or in a separate cell
!tensorboard --logdir runs --port 6006 --host 0.0.0.0
```

### Monitor Progress with TQDM

Training shows real-time progress bar:
```
Epoch 16/100: 100%|██████████| 156/156 [02:30<00:00, 1.03it/s, loss=1.23, pos_dist=0.45, neg_dist=0.67]
```

## Saving Results

### Checkpoints are saved automatically to:
```
/content/drive/MyDrive/checkpoints/
├── best_model.pt
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
└── ...
```

### Memory Bank saved to:
```
/content/drive/MyDrive/memory_bank/
└── memory_bank.pkl
```

### Download Results

```python
# Download specific file to local machine
from google.colab import files
files.download('/content/drive/MyDrive/checkpoints/best_model.pt')

# Or download entire folder
!zip -r /content/drive/MyDrive/checkpoints checkpoints.zip
files.download('checkpoints.zip')
```

## Colab-Specific Tips

### 1. Prevent Disconnection

Add this to keep your session alive:

```python
# Run in a separate cell to prevent timeout
import time
import torch

while True:
    try:
        # Dummy tensor operation
        torch.randn(1).cuda()
        time.sleep(60)  # Every minute
    except:
        break
```

Better: Install prevent-disconnect extension:
```python
%%javascript
// Run in browser console (F12)
function ClickConnect() {
    console.log("Working on keeping connected...")
    document.title = "Training in progress..."
    document.querySelector("#top-toolbar").style.background = "#e74c3c";  // Red = busy
    setTimeout(function() {
        document.title = "Colab Training";
        document.querySelector("#top-toolbar").style.background = "";
    }, 60000);
}
setInterval(ClickConnect, 120000);
```

### 2. Check GPU Availability

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
```

### 3. Clear GPU Memory if OOM

```python
import torch
import gc

# Clear cache
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared")
```

### 4. Handle File Paths

**Always use paths relative to Google Drive:**

```python
# GOOD - Relative paths
data_dir = "./alpha_prices_csv"
checkpoint_dir = "/content/drive/MyDrive/checkpoints"

# BAD - Absolute paths (breaks on Colab)
data_dir = "/home/user/alpha_prices_csv"
```

## Common Issues & Solutions

### Issue: "No assets loaded!"

**Problem:** CSV files not found

**Solution:**
```python
# Verify path
!ls /content/drive/MyDrive/alpha_prices_csv/ | head -5

# Or update my_config.py with correct path
# data_dir = "/content/drive/MyDrive/alpha_prices_csv"
```

### Issue: "CUDA out of memory"

**Solutions:**

**Option 1: Reduce batch size**
```python
# In my_config.py
training_config = TrainingConfig(
    batch_size = 128,  # Reduce from 256
)
```

**Option 2: Reduce lookback window**
```python
data_config = DataConfig(
    lookback_window = 72,  # Reduce from 168
    future_window = 24,
)
```

**Option 3: Use gradient accumulation**
```python
training_config = TrainingConfig(
    batch_size = 256,
    accumulation_steps = 2,  # Effective batch size = 512
)
```

### Issue: Training slows down after many epochs

**Problem:** Learning rate too low or model stuck in local minima

**Solution:**
```python
# Load checkpoint with higher learning rate
# In my_config.py:
training_config = TrainingConfig(
    learning_rate = 1e-4,  # Increase LR when resuming
)

# Then resume:
!python main.py \
    --config my_config.py \
    --checkpoint /content/drive/MyDrive/checkpoints/checkpoint_epoch_25.pt \
    --mode train
```

### Issue: Processed data not found

**Problem:** Wrong data_dir path

**Solution:**
```python
# Delete and regenerate
!rm -rf /content/drive/MyDrive/processed_data/

# Or check config matches data
# In my_config.py:
data_dir = "/content/drive/MyDrive/alpha_prices_csv"  # Full path
```

## Advanced: Custom Notebook

### Create `colab_training.ipynb`

```python
# Cell 1: Setup
from google.colab import drive
import os
import sys

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/timeseries-encoder')

# Cell 2: Install dependencies
!pip install torch numpy pandas tensorboard tqdm -q

# Cell 3: Import and verify
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Update config for Colab paths
config_text = """
from training.config import *

# Override paths for Colab
data_config = DataConfig(
    data_dir="/content/drive/MyDrive/alpha_prices_csv",
    lookback_window=168,
    future_window=24,
)

model_config = ModelConfig(
    input_dim=2,
    hidden_dims=(64, 128, 256),
)

training_config = TrainingConfig(
    num_epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    device="cuda",
    checkpoint_dir="/content/drive/MyDrive/checkpoints",
)

config = Config(
    data=data_config,
    model=model_config,
    training=training_config,
)
"""

with open('colab_config.py', 'w') as f:
    f.write(config_text)

# Cell 5: Train
!python main.py --config colab_config.py --mode full
```

## Batch Training (For Long Training)

### Strategy: Train in Chunks

If training takes >12 hours (Colab timeout):

**Chunk 1:** Epochs 0-20
```python
# In my_config.py
training_config = TrainingConfig(
    num_epochs=20,  # Train 20 epochs at a time
)
```

**Chunk 2:** Resume for epochs 20-40
```python
!python main.py \
    --config my_config.py \
    --checkpoint /content/drive/MyDrive/checkpoints/checkpoint_epoch_20.pt \
    --mode train
```

**Repeat** until desired epochs completed.

## Free Tier Limitations

**Colab Free:**
- GPU: K80 or T4 (16GB RAM)
- Max session: 12 hours (disconnect)
- Storage: 15GB Google Drive
- No guarantee of GPU availability

**Workarounds:**
- Save checkpoints frequently (every 5 epochs)
- Download checkpoints after each session
- Use prevent-disconnect script above
- Train in chunks (20 epochs per session)

## Performance Tips

### 1. Optimize DataLoader

```python
# In my_config.py
training_config = TrainingConfig(
    batch_size = 256,        # Optimize for your GPU
    num_workers = 4,         # Colab has multiple CPU cores
    pin_memory = True,
)
```

### 2. Use Mixed Precision (EXPERIMENTAL)

```python
# Add to main.py before training:
from torch.cuda.amp import autocast

# In train_step:
with autocast():
    # Forward pass in FP16
    anchor_emb, vol_emb = model(anchor_feat)
```

### 3. Profile Training

```python
# Add to main.py
import time

start = time.time()
# ... training code ...
print(f"Training time: {time.time() - start:.1f}s")
```

## Summary

**Basic workflow:**
1. Upload CSVs to Google Drive
2. Upload my_config.py to Google Drive
3. Create Colab notebook
4. Mount Google Drive
5. Install dependencies
6. Train
7. Download checkpoints

**Key paths:**
```python
/data: /content/drive/MyDrive/alpha_prices_csv/
/checkpoints: /content/drive/MyDrive/checkpoints/
/memory_bank: /content/drive/MyDrive/memory_bank/
/processed_data: /content/drive/MyDrive/timeseries-encoder/processed_data/
```

Your model will be trained on free GPU with results saved directly to your Google Drive!
