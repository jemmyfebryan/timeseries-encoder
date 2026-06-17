# Quick Start Guide

This guide will help you get started with the Dual-Head Siamese Triplet TCN quickly.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare your data:**

Place your CSV files in the correct location:
```bash
mkdir -p alpha_prices_csv
# Copy your CSV files to alpha_prices_csv/ with naming: alpha_prices_netuid_1.csv, ..., alpha_prices_netuid_128.csv
```

Each CSV should have:
- `close`: Close price (required)
- `timestamp`: Datetime (optional)

## Quick Start Options

### Option 1: Run with Synthetic Data (Test Mode)

First, verify everything works with dummy data:

```bash
python test_pipeline.py
```

This will create synthetic data and run a minimal pipeline test.

### Option 2: Run Full Pipeline with Your Data

```bash
python main.py --mode full
```

This will:
1. Load your data
2. Train the model
3. Create memory bank
4. Evaluate on test set

### Option 3: Step-by-Step

**Train only:**
```bash
python main.py --mode train
```

**Inference only (requires trained model):**
```bash
python main.py --mode inference --checkpoint checkpoints/best_model.pt
```

## Monitoring Training

**View logs:**
```bash
tail -f training.log
```

**Launch TensorBoard:**
```bash
tensorboard --logdir runs
# Open http://localhost:6006 in your browser
```

## Configuration

**Use default config:**
```bash
python main.py --mode full
```

**Use custom config:**
```bash
python main.py --mode full --config example_config.py
```

**Modify specific parameters:**
```bash
python main.py --mode full \
    --data-dir /path/to/data \
    --seed 123 \
    --log-file my_run.log
```

## Expected Output Structure

```
timeseries-encoder/
├── alpha_prices_csv/          # Your input data
│   ├── alpha_prices_netuid_1.csv
│   ├── ...
│   └── alpha_prices_netuid_128.csv
├── checkpoints/               # Model checkpoints
│   ├── best_model.pt         # Best model
│   └── checkpoint_epoch_*.pt # Periodic checkpoints
├── memory_bank/              # Saved memory banks
│   └── memory_bank.pkl
├── runs/                     # TensorBoard logs
│   └── run_*/
└── training.log              # Training logs
```

## Troubleshooting

**"No assets loaded" error:**
- Check CSV paths match the pattern: `alpha_prices_netuid_#.csv`
- Verify CSV files have a "close" column

**"CUDA out of memory":**
- Reduce batch size in config
- Or force CPU: Modify config `device="cpu"`

**Poor performance:**
- Increase `n_triplets` for better triplet diversity
- Adjust triplet margins (`return_margin`, `volatility_margin`)
- Try different `num_bins` for binning strategy

## Next Steps

1. Review `example_config.py` for customization options
2. Check `README.md` for detailed documentation
3. Experiment with hyperparameters
4. Monitor TensorBoard for training dynamics

## Key Files

- `main.py`: Entry point
- `config.py`: All hyperparameters
- `data.py`: Data loading and preprocessing
- `model.py`: TCN architecture
- `train.py`: Training loop
- `inference.py`: K-NN memory bank inference
- `dataset.py`: Triplet mining

Happy training! 🚀
