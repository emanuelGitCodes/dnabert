#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import random
from typing import List, Dict, Tuple, Any

# Check and install required packages
try:
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertForMaskedLM,
        AdamW,
        get_cosine_schedule_with_warmup,
    )
    from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
    from Bio import SeqIO
    from tqdm import tqdm
except ImportError:
    import sys
    import subprocess

    print("Installing required packages...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "transformers",
            "numpy",
            "scikit-learn",
            "biopython",
            "tqdm",
        ]
    )

    # Now import them
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    from transformers import BertConfig, BertForSequenceClassification, AdamW
    from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
    from Bio import SeqIO
    from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Check if MPS (Apple Silicon GPU) is available and set device accordingly
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
logger.info(f"Using device: {device}")

# ============================
# Data Processing Functions
# ============================


def load_fasta(
    fasta_file: str, max_sequences: int = None, sample_percentage: int = 100
) -> List[Tuple[str, str]]:
    """
    Load sequences from a FASTA file.

    Args:
        fasta_file: Path to the FASTA file
        max_sequences: Maximum number of sequences to load (for testing/development)
        sample_percentage: Percentage of sequences to randomly sample (default: 100)

    Returns:
        List of tuples (sequence_id, sequence)
    """
    logger.info(f"Loading sequences from {fasta_file}")

    # First count total sequences if sampling is needed
    if sample_percentage < 100:
        total_sequences = 0
        for _ in SeqIO.parse(fasta_file, "fasta"):
            total_sequences += 1

        if max_sequences and total_sequences > max_sequences:
            total_sequences = max_sequences

        num_to_sample = int(total_sequences * sample_percentage / 100)
        logger.info(
            f"Found {total_sequences} sequences, sampling {num_to_sample} ({sample_percentage}%)"
        )

        # Read all sequences (up to max_sequences) and randomly sample
        all_sequences = []
        for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
            if max_sequences and i >= max_sequences:
                break
            all_sequences.append((record.id, str(record.seq).upper()))

        # Randomly sample the specified percentage
        sampled_indices = random.sample(range(len(all_sequences)), num_to_sample)
        sequences = [all_sequences[i] for i in sampled_indices]

    else:
        # Original implementation when not sampling
        sequences = []
        try:
            for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
                if max_sequences and i >= max_sequences:
                    break
                sequences.append((record.id, str(record.seq).upper()))
        except Exception as e:
            logger.error(f"Error loading FASTA file: {e}")
            raise

    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


def generate_kmers(sequence: str, k: int = 6) -> List[str]:
    """
    Generate k-mers from a DNA sequence.

    Args:
        sequence: DNA sequence
        k: k-mer length (default: 6, which is common for DNABERT)

    Returns:
        List of k-mers
    """
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def process_sequences_to_kmers(
    sequences: List[Tuple[str, str]],
    k: int = 6,
    stride: int = 1,
    max_seq_length: int = 512,
    min_seq_length: int = 50,
) -> List[Dict[str, Any]]:
    """
    Process DNA sequences into k-mer format suitable for BERT.

    Args:
        sequences: List of (id, sequence) tuples
        k: k-mer size
        stride: Stride for k-mer generation
        max_seq_length: Maximum sequence length to process
        min_seq_length: Minimum sequence length to process

    Returns:
        List of processed sequences with their metadata
    """
    logger.info(f"Processing sequences to {k}-mers with stride {stride}")
    processed_data = []

    for seq_id, sequence in tqdm(sequences, desc="Processing sequences"):
        # Skip sequences that are too short
        if len(sequence) < min_seq_length:
            continue

        # Clean the sequence (keep only ATGC)
        clean_seq = "".join(c for c in sequence if c in "ATGC")

        # Skip if the sequence becomes too short after cleaning
        if len(clean_seq) < min_seq_length:
            continue

        # Generate k-mers
        for i in range(0, len(clean_seq) - min_seq_length, stride * max_seq_length):
            # Extract a chunk of the sequence
            chunk = clean_seq[i : i + max_seq_length]

            # Skip if chunk is too short
            if len(chunk) < min_seq_length:
                continue

            # Generate k-mers for this chunk
            kmers = generate_kmers(chunk, k)

            # Create an entry for this chunk
            processed_data.append(
                {
                    "id": f"{seq_id}_{i}",
                    "sequence": chunk,
                    "kmers": kmers,
                    "kmer_text": " ".join(kmers),  # Format for BERT tokenizer
                }
            )

    logger.info(f"Created {len(processed_data)} processed sequences")
    return processed_data


# ============================
# DNABERT Model
# ============================


class DNABertConfig:
    """Configuration for the DNABERT model"""

    def __init__(
        self,
        vocab_size: int = 4**6 + 5,  # For 6-mers (4^6) + special tokens
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        kmer_length: int = 6,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.kmer_length = kmer_length


class DNATokenizer:
    """
    Custom tokenizer for DNA sequences based on k-mers.
    This is a simplified version for illustration. For production, consider using
    a more sophisticated tokenizer or the pre-trained DNABERT tokenizer.
    """

    def __init__(self, kmer_length: int = 6):
        self.kmer_length = kmer_length

        # Generate vocabulary based on k-mer length
        nucleotides = ["A", "T", "G", "C"]
        self.vocab = {}

        # Add special tokens
        self.vocab["[PAD]"] = 0
        self.vocab["[UNK]"] = 1
        self.vocab["[CLS]"] = 2
        self.vocab["[SEP]"] = 3
        self.vocab["[MASK]"] = 4

        # Generate all possible k-mers for the vocabulary
        # In a real implementation, this would be more efficient
        # This is simplified for illustration
        counter = 5  # Start after special tokens
        for kmer in self._generate_all_kmers(nucleotides, kmer_length):
            self.vocab[kmer] = counter
            counter += 1

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def _generate_all_kmers(self, nucleotides, k, prefix=""):
        """Recursively generate all possible k-mers"""
        if k == 0:
            return [prefix]
        result = []
        for nucleotide in nucleotides:
            result.extend(
                self._generate_all_kmers(nucleotides, k - 1, prefix + nucleotide)
            )
        return result

    def encode(
        self,
        kmer_text: str,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ):
        """
        Encode k-mer text into token IDs.

        Args:
            kmer_text: Space-separated k-mers
            max_length: Maximum sequence length
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length

        Returns:
            Dictionary with input_ids, attention_mask, and token_type_ids
        """
        kmers = kmer_text.split()

        # Add special tokens
        kmers = ["[CLS]"] + kmers + ["[SEP]"]

        # Truncate if necessary
        if truncation and len(kmers) > max_length:
            kmers = kmers[: max_length - 1] + ["[SEP]"]

        # Convert to IDs
        input_ids = [self.vocab.get(kmer, self.vocab["[UNK]"]) for kmer in kmers]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad if necessary
        if padding and len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.vocab["[PAD]"]] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # Create token type IDs (all 0 for single sequence)
        token_type_ids = [0] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

    def batch_encode(
        self,
        kmer_texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ):
        """Encode a batch of k-mer texts"""
        batch_encodings = [
            self.encode(text, max_length, padding, truncation) for text in kmer_texts
        ]

        return {
            "input_ids": torch.stack(
                [encoding["input_ids"] for encoding in batch_encodings]
            ),
            "attention_mask": torch.stack(
                [encoding["attention_mask"] for encoding in batch_encodings]
            ),
            "token_type_ids": torch.stack(
                [encoding["token_type_ids"] for encoding in batch_encodings]
            ),
        }


def get_dnabert_config(kmer_length: int = 6) -> BertConfig:
    """
    Get a BertConfig object for DNABERT.

    Args:
        kmer_length: Length of k-mers

    Returns:
        BertConfig object
    """
    config = DNABertConfig(kmer_length=kmer_length)

    # Convert to a HuggingFace BertConfig
    hf_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        max_position_embeddings=config.max_position_embeddings,
        type_vocab_size=config.type_vocab_size,
        initializer_range=config.initializer_range,
        layer_norm_eps=config.layer_norm_eps,
    )

    return hf_config


def create_dnabert_model(
    kmer_length: int = 6, num_labels: int = 2
) -> BertForSequenceClassification:
    """
    Create a DNABERT model for sequence classification.

    Args:
        kmer_length: Length of k-mers
        num_labels: Number of output labels

    Returns:
        BertForSequenceClassification model
    """
    config = get_dnabert_config(kmer_length)

    # Create model with the config
    model = BertForSequenceClassification(config)

    # Initialize the model with random weights
    # In a real implementation, you might want to initialize from a pre-trained model
    # or use a pre-trained DNABERT model

    return model


# ============================
# Dataset and DataLoader
# ============================


class DNASequenceDataset(Dataset):
    """Dataset for DNA sequences"""

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode(
            item["kmer_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # For pre-training, create masked input
        if "labels" in item:
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "token_type_ids": encoding["token_type_ids"],
                "labels": item["labels"],
            }
        else:
            # For self-supervised pre-training
            return encoding


class MaskedDNADataset(DNASequenceDataset):
    """Dataset for masked language modeling with DNA sequences"""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        curriculum_masking: bool = False,
        current_epoch: int = 0,
        total_epochs: int = 3,
    ):
        super().__init__(data, tokenizer, max_length)
        self.mlm_probability = mlm_probability
        self.curriculum_masking = curriculum_masking
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode(
            item["kmer_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Create masked input for MLM
        input_ids = encoding["input_ids"].clone()
        labels = input_ids.clone()

        # Special tokens mask (don't mask special tokens)
        special_tokens_mask = [
            (
                1
                if token
                in [
                    self.tokenizer.vocab["[PAD]"],
                    self.tokenizer.vocab["[CLS]"],
                    self.tokenizer.vocab["[SEP]"],
                ]
                else 0
            )
            for token in input_ids.tolist()
        ]

        # Select tokens to mask (with curriculum masking if enabled)
        if self.curriculum_masking and self.total_epochs > 0:
            # Gradually increase masking probability throughout training
            # Early epochs: lower masking, later epochs: higher masking
            epoch_fraction = self.current_epoch / self.total_epochs
            # Scale masking probability from 0.15 to 0.25 as training progresses
            adjusted_mlm_probability = 0.15 + 0.1 * epoch_fraction
            logger.debug(
                f"Curriculum masking: epoch {self.current_epoch}, probability {adjusted_mlm_probability:.4f}"
            )
            probability_matrix = torch.full(labels.shape, adjusted_mlm_probability)
        else:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set the labels for unmasked tokens to -100 (ignored by loss function)
        labels[~masked_indices] = -100

        # Replace masked tokens with [MASK] token
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.vocab["[MASK]"]

        # Replace some masked tokens with random tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.1)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            5, len(self.tokenizer.vocab), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"],
            "labels": labels,
        }


# ============================
# Training and Evaluation Functions
# ============================


def train_model(
    model,
    train_dataloader,
    eval_dataloader=None,
    num_epochs=3,
    learning_rate=2e-5,
    warmup_steps=0,
    weight_decay=0.01,
    output_dir="./outputs",
    save_every=1,
    mixed_precision=True,
    gradient_accumulation_steps=1,
    early_stopping=False,
    patience=3,
    curriculum_masking=False,
):
    """
    Train the DNABERT model.

    Args:
        model: DNABERT model
        train_dataloader: DataLoader for training data
        eval_dataloader: Optional DataLoader for evaluation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for the learning rate scheduler
        weight_decay: Weight decay for AdamW optimizer
        output_dir: Directory to save model checkpoints
        save_every: Save model checkpoint every this many epochs
        mixed_precision: Whether to use mixed precision training

    Returns:
        Trained model
    """
    total_steps = len(train_dataloader) * num_epochs

    # Prepare optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up mixed precision training if available
    scaler = None
    if mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        elif (
            hasattr(torch, "amp")
            and hasattr(torch.amp, "autocast")
            and device.type != "mps"
        ):
            # For other platforms that support AMP but might not have CUDA
            scaler = torch.amp.GradScaler()
        elif device.type == "mps":
            logger.warning(
                "Mixed precision training not fully supported on MPS. Using MPS-specific optimizations instead."
            )
            # MPS-specific optimizations could be added here
        else:
            logger.warning("Mixed precision training not available on this device.")

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {total_steps}")
    global_step = 0
    model.zero_grad()

    # Early stopping variables
    best_eval_loss = float("inf")
    no_improvement_count = 0

    # Update dataset's epoch counter if using curriculum masking
    if curriculum_masking and hasattr(train_dataloader.dataset, "current_epoch"):
        train_dataloader.dataset.total_epochs = num_epochs

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0

        # Update dataset's epoch counter for curriculum masking
        if curriculum_masking and hasattr(train_dataloader.dataset, "current_epoch"):
            train_dataloader.dataset.current_epoch = epoch
        epoch_loss = 0

        # Training
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with or without mixed precision
            # Normalize loss for gradient accumulation
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            epoch_loss += (
                loss.item() * gradient_accumulation_steps
            )  # Undo normalization for logging

            # Backward pass with or without mixed precision
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer after accumulating gradients
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(
                train_dataloader
            ):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )  # Add gradient clipping for stability
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0
                    )  # Add gradient clipping for stability
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

            # Log progress
            if step % 100 == 0:
                logger.info(
                    f"  Step {step}/{len(train_dataloader)} - Loss: {loss.item():.4f}"
                )

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"  Average epoch loss: {avg_epoch_loss:.4f}")
        # Evaluate if eval_dataloader is provided
        if eval_dataloader:
            eval_results = evaluate_model(model, eval_dataloader)
            current_eval_loss = eval_results["loss"]
            logger.info(
                f"  Evaluation summary: Loss={current_eval_loss:.4f}, Accuracy={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}"
            )

            # Check for early stopping
            if early_stopping:
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    no_improvement_count = 0
                    # Save best model
                    save_model(model, f"{output_dir}/best-model")
                    logger.info(
                        f"  New best model saved with validation loss: {best_eval_loss:.4f}"
                    )
                else:
                    no_improvement_count += 1
                    logger.info(
                        f"  No improvement in validation loss for {no_improvement_count} epochs."
                    )
                    if no_improvement_count >= patience:
                        logger.info(
                            f"  Early stopping triggered after {epoch+1} epochs"
                        )
                        break

        # Save model checkpoint
        if (epoch + 1) % save_every == 0:
            save_model(model, f"{output_dir}/checkpoint-epoch-{epoch+1}")
            save_model(model, f"{output_dir}/checkpoint-epoch-{epoch+1}")

    # Save final model
    save_model(model, f"{output_dir}/final-model")

    return model


def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    Plot a confusion matrix using matplotlib and seaborn.

    Args:
        cm: Confusion matrix
        classes: List of class names
        title: Title for the plot
        cmap: Color map for the plot

    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save to file
    plt.tight_layout()
    confusion_matrix_file = "confusion_matrix.png"
    plt.savefig(confusion_matrix_file)
    logger.info(f"Confusion matrix saved to {confusion_matrix_file}")

    return plt.gcf()


def evaluate_model(model, eval_dataloader):
    """
    Evaluate the DNABERT model.

    Args:
        model: DNABERT model (either for classification or masked language modeling)
        eval_dataloader: DataLoader for evaluation data

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    eval_loss = 0
    all_preds = []
    all_labels = []

    # Check if this is a masked language modeling task
    is_mlm_task = isinstance(model, BertForMaskedLM)

    if is_mlm_task:
        logger.info("Evaluating with Masked Language Modeling (MLM) metrics")

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()

            # Get logits and labels
            logits = outputs.logits
            labels = batch["labels"]

            if is_mlm_task:
                # For MLM: logits shape is [batch_size, seq_length, vocab_size]
                # Reshape to [batch_size*seq_length, vocab_size]
                batch_size, seq_length, vocab_size = logits.shape
                logits = logits.view(-1, vocab_size)

                # Reshape labels to [batch_size*seq_length]
                labels = labels.view(-1)

                # Create mask for positions that were actually masked (where labels != -100)
                mask = labels != -100

                # Apply mask to keep only the predictions and labels for masked tokens
                masked_logits = logits[mask]
                masked_labels = labels[mask]

                # Get predictions for masked tokens only
                masked_preds = torch.argmax(masked_logits, dim=1).cpu().numpy()
                masked_labels = masked_labels.cpu().numpy()

                # Collect predictions and true labels for masked tokens
                all_preds.extend(masked_preds)
                all_labels.extend(masked_labels)
            else:
                # For classification tasks, use the original logic
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()

                # Collect predictions and true labels
                all_preds.extend(preds)
                all_labels.extend(labels)

    avg_eval_loss = eval_loss / len(eval_dataloader)

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics with 'weighted' average to handle potential imbalanced classes
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Log results
    logger.info(f"Evaluation results:")
    logger.info(f"  Loss: {avg_eval_loss:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")

    # For classification tasks (not MLM), calculate and plot confusion matrix
    if not is_mlm_task:
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"  Confusion Matrix:\n{cm}")

        # Get unique classes to determine if it's binary or multi-class
        unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
        num_classes = len(unique_classes)

        # Generate class labels
        class_labels = [str(i) for i in range(num_classes)]

        # Plot confusion matrix
        plot_confusion_matrix(cm, class_labels, title=f"Confusion Matrix")

        return {
            "loss": avg_eval_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }
    else:
        # For MLM tasks, return metrics without confusion matrix
        return {
            "loss": avg_eval_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def pretrain_model_mlm(
    model,
    train_dataloader,
    eval_dataloader=None,
    num_epochs=3,
    learning_rate=2e-5,
    warmup_steps=0,
    weight_decay=0.01,
    output_dir="./pretrained",
    save_every=1,
    mixed_precision=True,
    gradient_accumulation_steps=1,
    early_stopping=False,
    patience=3,
    curriculum_masking=False,
):
    """
    Pre-train the DNABERT model using masked language modeling.

    This function is similar to train_model but specifically for pre-training
    with masked language modeling objective.

    Args:
        model: DNABERT model configured for masked language modeling
        train_dataloader: DataLoader for masked training data
        eval_dataloader: Optional DataLoader for masked evaluation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps for the learning rate scheduler
        weight_decay: Weight decay for AdamW optimizer
        output_dir: Directory to save model checkpoints
        save_every: Save model checkpoint every this many epochs
        mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients before performing a backward/update pass
        early_stopping: Whether to use early stopping based on validation loss
        patience: Number of epochs with no improvement after which training will be stopped if early_stopping is enabled
        curriculum_masking: Whether to use curriculum masking strategy that increases masking probability during training

    Returns:
        Pre-trained model
    """
    # This implementation is similar to train_model but with MLM-specific logging
    return train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_dir=output_dir,
        save_every=save_every,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        early_stopping=early_stopping,
        patience=patience,
        curriculum_masking=curriculum_masking,
    )


# ============================
# Model Saving and Loading
# ============================


def save_model(model, output_dir):
    """
    Save a DNABERT model to disk.

    Args:
        model: DNABERT model
        output_dir: Directory to save the model
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Saving model to {output_dir}")

    # Save model
    model.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir}")


def load_model(model_path, num_labels=2):
    """
    Load a DNABERT model from disk.

    Args:
        model_path: Path to the saved model
        num_labels: Number of output labels

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )

    # Move model to device
    model.to(device)

    logger.info(f"Model loaded from {model_path}")

    return model


# ============================
# Memory Optimization Functions
# ============================


def optimize_memory_usage(dataloader_kwargs=None):
    """
    Apply memory optimization techniques for M1 Max with 64GB RAM.

    Args:
        dataloader_kwargs: Additional kwargs for DataLoader

    Returns:
        Dictionary of optimized DataLoader kwargs
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # Default optimization settings
    optimized_kwargs = {
        "pin_memory": True,
        "num_workers": 4,  # Use multiple workers for data loading
        "prefetch_factor": 2,  # Prefetch batches
    }

    # Update with user-provided kwargs
    optimized_kwargs.update(dataloader_kwargs)

    # Set environment variables for PyTorch memory efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    return optimized_kwargs


def create_optimized_dataloader(dataset, batch_size=32, shuffle=True, **kwargs):
    """
    Create a memory-optimized DataLoader.

    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        **kwargs: Additional kwargs for DataLoader

    Returns:
        Optimized DataLoader
    """
    # Get optimized kwargs
    dataloader_kwargs = optimize_memory_usage(kwargs)

    # Create sampler based on shuffle parameter
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **dataloader_kwargs,
    )

    return dataloader


def chunk_and_process_dataset(
    fasta_file, chunk_size=1000, max_sequences=None, k=6, **kwargs
):
    """
    Process a large FASTA file in chunks to conserve memory.

    Args:
        fasta_file: Path to FASTA file
        chunk_size: Number of sequences to process at once
        max_sequences: Maximum number of sequences to process
        k: k-mer size
        **kwargs: Additional kwargs for process_sequences_to_kmers

    Returns:
        List of processed sequences
    """
    logger.info(f"Processing {fasta_file} in chunks of {chunk_size}")

    all_processed_data = []
    seq_count = 0

    # Process file in chunks
    chunk = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        if max_sequences and seq_count >= max_sequences:
            break

        chunk.append((record.id, str(record.seq).upper()))
        seq_count += 1

        # Process chunk when it reaches chunk_size
        if len(chunk) >= chunk_size:
            processed_chunk = process_sequences_to_kmers(chunk, k=k, **kwargs)
            all_processed_data.extend(processed_chunk)
            chunk = []

            # Free memory
            import gc

            gc.collect()

    # Process any remaining sequences
    if chunk:
        processed_chunk = process_sequences_to_kmers(chunk, k=k, **kwargs)
        all_processed_data.extend(processed_chunk)

    logger.info(f"Processed {len(all_processed_data)} sequences in total")

    return all_processed_data


# ============================
# Main Execution
# ============================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate DNABERT models.")

    parser.add_argument(
        "--fasta_file",
        type=str,
        required=True,
        help="Path to the FASTA file containing genomic sequences.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "train", "eval"],
        help="Mode: pretrain (self-supervised learning), train (supervised learning), or eval (evaluate model)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model checkpoints and results",
    )

    parser.add_argument(
        "--kmer",
        type=int,
        default=6,
        help="k-mer length (default: 6)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for processing",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimization",
    )

    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process (for testing/development)",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of sequences to process at once (for memory efficiency)",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="Stride for sequence chunking in base pairs",
    )

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking tokens in masked language modeling",
    )

    parser.add_argument(
        "--sample_percentage",
        type=int,
        default=100,
        help="Percentage of sequences to randomly sample from the input FASTA file",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training if available",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before performing a backward/update pass",
    )

    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping based on validation loss",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement after which training will be stopped",
    )

    parser.add_argument(
        "--curriculum_masking",
        action="store_true",
        help="Enable curriculum masking with gradually increasing masking probability",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save model checkpoint every this many epochs",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to load a pre-trained model (for train or eval modes)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging to file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"Arguments: {args}")
    logger.info(f"Device: {device}")

    # Process data
    logger.info("Processing genomic data...")

    if args.mode == "pretrain" or args.mode == "train":
        # Process data in chunks to save memory
        sequences = load_fasta(
            args.fasta_file,
            max_sequences=args.max_sequences,
            sample_percentage=args.sample_percentage,
        )

        processed_data = process_sequences_to_kmers(
            sequences,
            k=args.kmer,
            stride=args.stride,
            max_seq_length=args.max_seq_length,
        )

        # Create tokenizer
        tokenizer = DNATokenizer(kmer_length=args.kmer)

        # Split data for training and evaluation
        train_data, eval_data = train_test_split(
            processed_data, test_size=0.1, random_state=args.seed
        )

        logger.info(f"Train data size: {len(train_data)}")
        logger.info(f"Eval data size: {len(eval_data)}")

        if args.mode == "pretrain":
            # Create datasets for pre-training with masked language modeling
            train_dataset = MaskedDNADataset(
                train_data,
                tokenizer,
                max_length=args.max_seq_length,
                mlm_probability=args.mlm_probability,
                curriculum_masking=args.curriculum_masking,
                current_epoch=0,
                total_epochs=args.num_epochs,
            )
            eval_dataset = MaskedDNADataset(
                eval_data,
                tokenizer,
                max_length=args.max_seq_length,
                mlm_probability=args.mlm_probability,
                curriculum_masking=args.curriculum_masking,
                current_epoch=0,
                total_epochs=args.num_epochs,
            )

            # Create optimized data loaders
            train_dataloader = create_optimized_dataloader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            eval_dataloader = create_optimized_dataloader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )

            # Create model for pre-training
            from transformers import BertForMaskedLM

            config = get_dnabert_config(kmer_length=args.kmer)
            model = BertForMaskedLM(config)
            model.to(device)

            # Pre-train the model
            pretrain_model_mlm(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                save_every=args.save_every,
                mixed_precision=args.mixed_precision,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                early_stopping=args.early_stopping,
                patience=args.patience,
                curriculum_masking=args.curriculum_masking,
            )

            logger.info("Pre-training complete!")

        elif args.mode == "train":
            # For training mode, we assume we're fine-tuning for a classification task
            # Create datasets for classification
            # First, add dummy labels for demonstration (in a real task, you'd use actual labels)
            for item in train_data:
                item["labels"] = random.randint(
                    0, 1
                )  # Binary classification for demonstration
            for item in eval_data:
                item["labels"] = random.randint(0, 1)

            train_dataset = DNASequenceDataset(
                train_data, tokenizer, max_length=args.max_seq_length
            )
            eval_dataset = DNASequenceDataset(
                eval_data, tokenizer, max_length=args.max_seq_length
            )

            # Create optimized data loaders
            train_dataloader = create_optimized_dataloader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            eval_dataloader = create_optimized_dataloader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
            )

            # Create or load model
            if args.model_path:
                # Load pre-trained model
                model = load_model(
                    args.model_path, num_labels=2
                )  # Binary classification
            else:
                # Create new model
                model = create_dnabert_model(kmer_length=args.kmer, num_labels=2)
                model.to(device)

            # Train the model
            train_model(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                output_dir=args.output_dir,
                save_every=args.save_every,
                mixed_precision=args.mixed_precision,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                early_stopping=args.early_stopping,
                patience=args.patience,
            )

            logger.info("Training complete!")

    elif args.mode == "eval":
        # Evaluation mode requires a pre-trained model
        if not args.model_path:
            logger.error("Model path must be provided for evaluation mode")
            return

        # Process data for evaluation
        # Process data for evaluation
        sequences = load_fasta(
            args.fasta_file,
            max_sequences=args.max_sequences,
            sample_percentage=args.sample_percentage,
        )

        processed_data = process_sequences_to_kmers(
            sequences,
            k=args.kmer,
            stride=args.stride,
            max_seq_length=args.max_seq_length,
        )
        # Create tokenizer
        tokenizer = DNATokenizer(kmer_length=args.kmer)

        # For demonstration, add dummy labels
        for item in processed_data:
            item["labels"] = random.randint(0, 1)

        # Create dataset
        eval_dataset = DNASequenceDataset(
            processed_data, tokenizer, max_length=args.max_seq_length
        )

        # Create data loader
        eval_dataloader = create_optimized_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        # Load model
        model = load_model(args.model_path, num_labels=2)

        # Evaluate model
        eval_results = evaluate_model(model, eval_dataloader)

        logger.info(
            f"Evaluation summary: Loss={eval_results['loss']:.4f}, Accuracy={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}"
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
