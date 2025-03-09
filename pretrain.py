import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

from prepare_datasets import prepare_dataloaders
from vqa_model import MedicalVQAModel, train_vqa_model
from data_preprocessing import create_question_mask

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
BASE_DIR = "e:/MUMC_v2"
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints", "pretrain")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create necessary directories
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def pretrain_model(
    batch_size=64,
    num_epochs=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=1000,
    contrastive_weight=0.9,  # Higher weight for contrastive loss during pretraining
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Pretrain the Medical VQA model using contrastive learning.
    
    Args:
        batch_size: Batch size for training
        num_epochs: Number of pretraining epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        contrastive_weight: Weight for contrastive loss component
        device: Device to train on
    """
    logger.info("Starting contrastive pretraining...")
    logger.info(f"Using device: {device}")
    
    # Prepare dataloaders
    train_loader, val_loader, _, preprocessor = prepare_dataloaders(batch_size=batch_size)
    
    # Get vocabulary size
    vocab_size = len(preprocessor.word2idx)
    num_answers = len(preprocessor.answer2idx)
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Number of unique answers: {num_answers}")
    
    # Initialize model
    model = MedicalVQAModel(
        # Vision parameters
        img_size=224,
        patch_size=16,
        vision_embed_dim=768,
        vision_depth=12,
        vision_num_heads=12,
        # Language parameters
        vocab_size=vocab_size,
        max_seq_len=preprocessor.max_seq_len,
        language_embed_dim=768,
        language_depth=6,  # Reduced depth for faster pretraining
        language_num_heads=12,
        # Fusion parameters
        fusion_dim=768,
        fusion_method="cross_attention",
        # Answer prediction parameters
        num_open_answers=num_answers,
        # Masking parameters
        use_cluster_mask=True,
        anchor_ratio=0.05,
        similarity_threshold=0.75,
        min_mask_ratio=0.5,
        # Contrastive learning parameters
        contrastive_proj_dim=256,
        use_contrastive=True,
        # Other parameters
        dropout=0.1,
    )
    
    # Train model with pretraining mode enabled
    train_vqa_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        contrastive_weight=contrastive_weight,
        pretraining_mode=True,  # Enable pretraining mode
        device=device,
        save_path=CHECKPOINTS_DIR,
        log_interval=50,
    )
    
    # Save final pretrained model
    pretrained_path = os.path.join(CHECKPOINTS_DIR, "pretrained_final.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "num_answers": num_answers,
    }, pretrained_path)
    
    logger.info(f"Pretraining complete. Model saved to {pretrained_path}")
    
    return model, pretrained_path

def plot_pretraining_metrics(log_file):
    """
    Plot pretraining metrics from log file.
    
    Args:
        log_file: Path to log file
    """
    # Extract metrics from log file
    epochs = []
    losses = []
    contrastive_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch:" in line and "Loss:" in line and "Contrastive:" in line:
                parts = line.split('|')
                epoch_part = parts[0].strip()
                loss_part = [p for p in parts if "Loss:" in p][0].strip()
                contrastive_part = [p for p in parts if "Contrastive:" in p][0].strip()
                
                epoch = float(epoch_part.split(':')[1].split('/')[0].strip())
                loss = float(loss_part.split(':')[1].strip())
                contrastive_loss = float(contrastive_part.split(':')[1].strip())
                
                epochs.append(epoch)
                losses.append(loss)
                contrastive_losses.append(contrastive_loss)
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'b-')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, contrastive_losses, 'r-')
    plt.title('Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, 'pretraining_metrics.png'))
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Configure logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"pretrain_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Run pretraining
    model, pretrained_path = pretrain_model(
        batch_size=64,
        num_epochs=5,  # Fewer epochs for pretraining
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        contrastive_weight=0.9,
    )
    
    # Plot pretraining metrics
    plot_pretraining_metrics(log_file)
    
    logger.info(f"Pretraining completed. Model saved to {pretrained_path}")