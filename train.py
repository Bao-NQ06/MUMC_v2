import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
from PIL import Image

from prepare_datasets import prepare_dataloaders
from vqa_model import MedicalVQAModel, train_vqa_model, validate_vqa_model
from data_preprocessing import create_question_mask

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
BASE_DIR = "e:/MUMC_v2"
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints", "train")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create necessary directories
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model(
    pretrained_path=None,
    batch_size=32,
    num_epochs=20,
    learning_rate=5e-5,  # Lower learning rate for fine-tuning
    weight_decay=0.01,
    warmup_steps=500,
    contrastive_weight=0.2,  # Lower weight for contrastive loss during fine-tuning
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train the Medical VQA model, optionally starting from a pretrained checkpoint.
    
    Args:
        pretrained_path: Path to pretrained model checkpoint (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        contrastive_weight: Weight for contrastive loss component
        device: Device to train on
    """
    logger.info("Starting model training...")
    logger.info(f"Using device: {device}")
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders(batch_size=batch_size)
    
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
        language_depth=12,
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
    
    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Train model
    train_vqa_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        contrastive_weight=contrastive_weight,
        pretraining_mode=False,  # Disable pretraining mode
        device=device,
        save_path=CHECKPOINTS_DIR,
        log_interval=50,
    )
    
    # Save final model
    final_path = os.path.join(CHECKPOINTS_DIR, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "num_answers": num_answers,
    }, final_path)
    
    logger.info(f"Training complete. Model saved to {final_path}")
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, preprocessor, device)
    
    return model, test_metrics

def evaluate_model(model, test_loader, preprocessor, device):
    """
    Evaluate the model on the test set and visualize results.
    
    Args:
        model: Trained MedicalVQAModel
        test_loader: DataLoader for test data
        preprocessor: VQAPreprocessor instance
        device: Device to evaluate on
        
    Returns:
        Dictionary of test metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Set up criterion
    criterion = nn.CrossEntropyLoss()
    
    # Get test metrics
    test_metrics = validate_vqa_model(model, test_loader, criterion, device)
    
    logger.info(f"Test metrics: {test_metrics}")
    
    # Visualize sample predictions
    visualize_predictions(model, test_loader, preprocessor, device)
    
    return test_metrics

def visualize_predictions(model, test_loader, preprocessor, device, num_samples=10):
    """
    Visualize model predictions compared to ground truth answers.
    
    Args:
        model: Trained MedicalVQAModel
        test_loader: DataLoader for test data
        preprocessor: VQAPreprocessor instance
        device: Device to evaluate on
        num_samples: Number of samples to visualize
    """
    logger.info(f"Visualizing {num_samples} sample predictions...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data
    all_samples = []
    with torch.no_grad():
        for batch in test_loader:
            # Get batch data
            images = batch["images"].to(device)
            question_tokens = batch["question_tokens"].to(device)
            question_mask = batch.get("question_mask", None)
            if question_mask is not None:
                question_mask = question_mask.to(device)
            answer_labels = batch["answer_labels"].to(device)
            question_texts = batch["question_texts"]
            answer_texts = batch["answer_texts"]
            
            # Forward pass
            outputs = model(images, question_tokens, question_mask, is_training=False)
            
            # Extract outputs
            question_type_logits = outputs['question_type_logits']
            open_answer_logits = outputs['open_answer_logits']
            yesno_logits = outputs['yesno_logits']
            
            # Get predicted question types
            question_type_preds = torch.argmax(question_type_logits, dim=1)  # 0=open, 1=yes/no
            
            # Get predicted answers
            for i in range(images.size(0)):
                # Determine if this is a yes/no question based on prediction
                is_yesno = question_type_preds[i].item() == 1
                
                if is_yesno:
                    # Get yes/no prediction
                    yesno_pred = torch.argmax(yesno_logits[i]).item()
                    pred_answer = "yes" if yesno_pred == 1 else "no"
                else:
                    # Get open-ended prediction
                    open_pred = torch.argmax(open_answer_logits[i]).item()
                    pred_answer = preprocessor.idx2answer[open_pred]
                
                # Get ground truth answer
                gt_answer = answer_texts[i]
                
                # Store sample data
                sample = {
                    "image": images[i].cpu(),
                    "question": question_texts[i],
                    "predicted_answer": pred_answer,
                    "ground_truth": gt_answer,
                    "correct": pred_answer.lower() == gt_answer.lower()
                }
                all_samples.append(sample)
                
                if len(all_samples) >= num_samples:
                    break
            
            if len(all_samples) >= num_samples:
                break
    
    # Create visualization
    fig, axes = plt.subplots(nrows=min(num_samples, len(all_samples)), ncols=1, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, sample in enumerate(all_samples[:num_samples]):
        # Get image
        img = sample["image"]
        img = img.permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Add text
        color = 'green' if sample["correct"] else 'red'
        axes[i].set_title(f"Q: {sample['question']}\nPredicted: {sample['predicted_answer']} | Ground Truth: {sample['ground_truth']}", 
                          color=color, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'))
    plt.close()
    
    # Save detailed results to JSON
    results = []
    for sample in all_samples:
        # Convert tensor to list for JSON serialization
        sample_copy = sample.copy()
        sample_copy.pop("image")
        results.append(sample_copy)
    
    with open(os.path.join(RESULTS_DIR, 'prediction_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate accuracy
    correct = sum(1 for sample in all_samples if sample["correct"])
    accuracy = correct / len(all_samples) if all_samples else 0
    logger.info(f"Sample accuracy: {accuracy:.4f} ({correct}/{len(all_samples)})")

def plot_training_metrics(log_file):
    """
    Plot training metrics from log file.
    
    Args:
        log_file: Path to log file
    """
    # Extract metrics from log file
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch:" in line and "Loss:" in line and not "Val" in line:
                parts = line.split('|')
                epoch_part = parts[0].strip()
                loss_part = [p for p in parts if "Loss:" in p][0].strip()
                
                epoch = float(epoch_part.split(':')[1].split('/')[0].strip())
                loss = float(loss_part.split(':')[1].strip())
                
                epochs.append(epoch)
                train_losses.append(loss)
            
            if "Val Loss:" in line and "Val Open Acc:" in line:
                parts = line.split('|')
                val_loss_part = [p for p in parts if "Val Loss:" in p][0].strip()
                val_open_acc_part = [p for p in parts if "Val Open Acc:" in p][0].strip()
                
                val_loss = float(val_loss_part.split(':')[1].strip())
                val_acc = float(val_open_acc_part.split(':')[1].strip())
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
    
    # Plot metrics
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if val_accs:
        plt.plot(epochs[:len(val_accs)], val_accs, 'g-', label='Val Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, 'training_metrics.png'))
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Configure logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"train_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Check if pretrained model exists
    pretrained_path = os.path.join(BASE_DIR, "checkpoints", "pretrain", "pretrained_final.pth")
    if os.path.exists(pretrained_path):
        logger.info(f"Found pretrained model at {pretrained_path}")
    else:
        logger.info("No pretrained model found. Training from scratch.")
        pretrained_path = None
    
    # Run training
    model, test_metrics = train_model(
        pretrained_path=pretrained_path,
        batch_size=32,
        num_epochs=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        contrastive_weight=0.2,
    )
    
    # Plot training metrics
    plot_training_metrics(log_file)
    
    logger.info(f"Training and evaluation completed. Results saved to {RESULTS_DIR}")