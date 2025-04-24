# DNABERT - Genomic Sequence Modeling

  ## Overview
  This implementation uses a BERT-style architecture for genomic sequence analysis. The code supports:
  - Self-supervised pre-training with Masked Language Modeling (MLM)
  - Supervised training for sequence classification
  - Model evaluation

  ## Features
  - k-mer tokenization of DNA sequences
  - Apple Silicon GPU (MPS) and CUDA support
  - Memory optimization for large datasets
  - Curriculum masking strategies
  - Multiple training modes

  ## Installation
  ```bash
  # Install required packages
  pip install torch transformers numpy scikit-learn biopython tqdm matplotlib seaborn
  ```

  ## Basic Usage

  ### Modes
  1. **Pretrain**: Self-supervised pre-training with Masked Language Modeling
  2. **Train**: Supervised training for classification tasks
  3. **Eval**: Model evaluation

  ### Command Line Parameters
  | Parameter | Type | Default | Description |
  |-----------|------|---------|-------------|
  | `--fasta_file` | str | **Required** | Path to the FASTA file containing genomic sequences |
  | `--mode` | str | "pretrain" | Mode: pretrain, train, or eval |
  | `--output_dir` | str | "./outputs" | Output directory for model checkpoints and results |
  | `--kmer` | int | 6 | k-mer length |
  | `--max_seq_length` | int | 512 | Maximum sequence length for processing |
  | `--batch_size` | int | 32 | Batch size for training and evaluation |
  | `--num_epochs` | int | 3 | Number of training epochs |
  | `--learning_rate` | float | 2e-5 | Learning rate for optimization |
  | `--max_sequences` | int | None | Maximum number of sequences to process |
  | `--chunk_size` | int | 1000 | Number of sequences to process at once for memory efficiency |
  | `--stride` | int | 50 | Stride for sequence chunking in base pairs |
  | `--mlm_probability` | float | 0.15 | Probability of masking tokens in masked language modeling |
  | `--sample_percentage` | int | 100 | Percentage of sequences to randomly sample |
  | `--mixed_precision` | flag | False | Use mixed precision training if available |
  | `--gradient_accumulation_steps` | int | 1 | Number of gradient accumulation steps |
  | `--early_stopping` | flag | False | Enable early stopping based on validation loss |
  | `--patience` | int | 3 | Number of epochs with no improvement for early stopping |
  | `--curriculum_masking` | flag | False | Enable curriculum masking |
  | `--save_every` | int | 1 | Save model checkpoint every this many epochs |
  | `--model_path` | str | None | Path to load a pre-trained model |
  | `--seed` | int | 42 | Random seed for reproducibility |

  ## Dataset
  - The dataset used for training and evaluation is the [NCBI Dataset Homo Sapiens](https://www.ncbi.nlm.nih.gov/datasets/taxonomy/9606/)
  ```bash
  GCA_000001405.28/GCA_000001405.28_GRCh38.p13_genomic.fna
  ```
  ## Example Commands

  **1. Pre-training:**
  ```bash
  python dnaBert.py \
    --fasta_file data/genomes.fasta \
    --mode pretrain \
    --output_dir pretrained_model \
    --kmer 6 \
    --mixed_precision
  ```

  **2. Supervised Training:**
  ```bash
  python dnaBert.py \
    --fasta_file training_data.fasta \
    --mode train \
    --model_path pretrained_model/final-model \
    --output_dir classifier_model \
    --num_epochs 10 \
    --early_stopping
  ```

  **3. Evaluation:**
  ```bash
  python dnaBert.py \
    --fasta_file test_data.fasta \
    --mode eval \
    --model_path classifier_model/final-model \
    --batch_size 64
  ```

  ## Outputs
  - Saved model checkpoints in specified output directory
  - Training logs (log.txt)
  - Confusion matrix visualization (confusion_matrix.png)

## Hardware optimization:
- Optimized for MacBook Pro with M1 Max and 64GB RAM
- Uses mixed precision training where possible
- Implements efficient data loading and processing

## Dependencies:
- PyTorch (with MPS acceleration for M1 Mac)
- Transformers (Hugging Face)
- NumPy
- scikit-learn
- Biopython (for FASTA parsing)
- tqdm (for progress bars)
