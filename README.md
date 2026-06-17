# Dual-Head Siamese Triplet TCN for Time-Series Representation Learning

A production-grade PyTorch implementation of a Dual-Head Siamese Triplet Temporal Convolutional Network (TCN) for learning representations from asset price time-series data.

## Overview

This system learns scale-invariant embeddings of time-series windows that capture both future return and volatility characteristics. It uses triplet learning with dual projection heads, enabling similarity-based search and prediction using a K-NN memory bank approach.

## Features

- **Dual-Head Architecture**: Separate embedding heads for return and volatility similarity
- **Triplet Learning**: Efficient binning-based triplet mining strategy
- **TCN Backbone**: Temporal Convolutional Network with dilated convolutions
- **K-NN Memory Bank**: Weighted K-NN inference with softmax similarity weighting
- **Production-Ready**: Complete training, validation, checkpointing, and inference pipeline
- **Robust Data Handling**: Per-asset train/val/test splitting to prevent look-ahead bias

## Architecture

```
Input (12-hour window)
    ↓
TCN Backbone (Dilated 1D CNN)
    ↓
Shared Representation
    ├→ Return Head → Return Embedding (32-dim)
    └→ Volatility Head → Volatility Embedding (32-dim)
```

## Installation

```bash
# Clone the repository
cd timeseries-encoder

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The system expects 128 CSV files with the following structure:

**Location**: `./alpha_prices_csv/alpha_prices_netuid_#.csv` (where `#` ranges from 1 to 128)

**Required columns**:
- `close`: Close price (required)
- `timestamp`: Datetime (optional, will use index if not provided)

**Example CSV**:
```csv
timestamp,close
2024-01-01 00:00:00,100.5
2024-01-01 01:00:00,101.2
2024-01-01 02:00:00,100.8
...
```

## Usage

### Basic Usage

```bash
# Full pipeline (training + inference)
python main.py --mode full

# Training only
python main.py --mode train

# Inference only (requires checkpoint)
python main.py --mode inference --checkpoint checkpoints/best_model.pt
```

### Advanced Usage

```bash
# Custom data directory
python main.py --mode full --data-dir /path/to/data

# Custom random seed
python main.py --mode full --seed 123

# Load custom configuration
python main.py --mode full --config my_config.py

# Custom log file
python main.py --mode full --log-file my_run.log

# Specify memory bank path
python main.py --mode inference --checkpoint checkpoints/best_model.pt --memory-bank-path memory_bank/mine.pkl
```

## Configuration

The system can be configured via a custom Python configuration file. Create a file (e.g., `my_config.py`) with a `config` or `Config` instance:

```python
from config import Config, DataConfig, ModelConfig, TripletConfig, TrainingConfig

# Create custom configurations
data_config = DataConfig(
    lookback_window=12,
    future_window=12,
    min_samples=100,
)

model_config = ModelConfig(
    hidden_dims=(64, 128, 256),
    embedding_dim=64,
    dropout=0.2,
)

triplet_config = TripletConfig(
    strategy="binning",
    num_bins=20,
    return_margin=1.0,
    volatility_margin=1.0,
    n_triplets=10000,
)

training_config = TrainingConfig(
    num_epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    early_stopping_patience=15,
)

# Combine into main config
config = Config(
    data=data_config,
    model=model_config,
    triplet=triplet_config,
    training=training_config,
)
```

Then run:
```bash
python main.py --config my_config.py
```

## Pipeline Stages

### 1. Data Loading & Preprocessing

- Loads 128 CSV files in parallel
- Computes log returns from close prices
- Creates sliding 12-hour windows
- Calculates targets (future accumulated return and volatility)
- Splits data per asset: 60% train, 20% val, 20% test

### 2. Triplet Mining

Uses efficient binning strategy:
- Bins targets into discrete quantiles
- Samples positives from same bin (similar targets)
- Samples negatives from different bins (dissimilar targets)
- Alternates between return and volatility similarity

### 3. Model Training

- **Loss**: Triplet Margin Loss for both heads
  - `Total Loss = Return_Triplet_Loss + Volatility_Triplet_Loss`
- **Optimizer**: Adam with learning rate scheduling
- **Validation**: Checked every epoch
- **Early Stopping**: Stops when validation loss doesn't improve
- **Checkpointing**: Saves best model and periodic checkpoints

### 4. Inference (K-NN Memory Bank)

1. Encode test set samples into embeddings (Memory Bank)
2. For a query window:
   - Compute embeddings
   - Find Top-K most similar historical states (cosine similarity)
   - Apply softmax to similarities → weights
   - Predict as weighted average: `Σ softmax(similarity_i) * target_i`

## Outputs

### Checkpoints
- `checkpoints/best_model.pt`: Best model (lowest validation loss)
- `checkpoints/checkpoint_epoch_N.pt`: Periodic checkpoints

### Memory Bank
- `memory_bank/memory_bank.pkl`: Serialized memory bank with embeddings and targets

### Logs
- `training.log`: Training logs
- `runs/run_TIMESTAMP/`: TensorBoard logs

### TensorBoard
```bash
tensorboard --logdir runs
```

## Model Architecture Details

### TCN Backbone
- **Layers**: 3 temporal blocks with increasing dilation (1, 2, 4)
- **Channels**: 64 → 128 → 256
- **Kernel size**: 3
- **Activation**: ReLU
- **Normalization**: Batch Normalization
- **Regularization**: Dropout (0.2)

### Projection Heads
- **Structure**: Linear → BN → ReLU → Dropout → Linear
- **Output dimensions**: 32-dim for both heads
- **Normalization**: L2 normalization for cosine similarity

## Key Design Choices

### Why TCN?
- Captures temporal patterns through dilated convolutions
- Efficient receptive field growth
- Parallel processing (unlike RNNs)
- Proven for time-series tasks

### Why Dual Heads?
- Return and volatility have different characteristics
- Separate heads can specialize on their respective tasks
- Enables targeted similarity search

### Why Triplet Learning?
- Learns similarity structure directly
- Scale-invariant representations
- Works well with downstream similarity-based prediction

### Why K-NN Memory Bank?
- Non-parametric prediction
- Interpretable (weighted average of similar cases)
- No need to train separate prediction heads
- Naturally handles uncertainty through softmax weights

## Performance Considerations

### Memory Usage
- Each embedding: 32-dim float32 = 128 bytes
- Memory bank with 100K samples: ~12.8 MB per head
- Total for both heads: ~25.6 MB (negligible)

### Training Speed
- Triplet mining: Pre-computed, ~1-2 minutes for full dataset
- Training speed: ~10-50 samples/second/GPU (depends on batch size and hardware)

### Inference Speed
- Single query: ~1-5ms (embedding + similarity search)
- Batch queries: Faster due to parallelization

## Troubleshooting

### Issue: "No assets loaded"
- Check that CSV files exist at the expected paths
- Verify CSV files have a "close" column
- Ensure files have enough samples (>100)

### Issue: "CUDA out of memory"
- Reduce `batch_size` in TrainingConfig
- Reduce `hidden_dims` in ModelConfig
- Use CPU by setting `device="cpu"`

### Issue: Poor validation loss
- Increase `n_triplets` for better triplet diversity
- Adjust `return_margin` and `volatility_margin`
- Try different `num_bins` for binning strategy
- Increase model capacity

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dual_head_siamese_tcn,
  title={Dual-Head Siamese Triplet TCN for Time-Series Representation Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/timeseries-encoder}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
