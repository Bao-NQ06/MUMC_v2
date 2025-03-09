import os
import argparse
import torch
import logging
from datetime import datetime

# Import project modules
from prepare_datasets import prepare_dataloaders
from pretrain import pretrain_model, plot_pretraining_metrics
from train import train_model, plot_training_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
BASE_DIR = "e:/MUMC_v2"
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create necessary directories
os.makedirs(os.path.join(CHECKPOINTS_DIR, "pretrain"), exist_ok=True)
os.makedirs(os.path.join(CHECKPOINTS_DIR, "train"), exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Medical VQA Training Pipeline")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        choices=["prepare", "pretrain", "train", "all"],
        help="Mode to run: prepare data, pretrain model, train model, or all steps"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--pretrain_epochs", 
        type=int, 
        default=5,
        help="Number of epochs for pretraining"
    )
    parser.add_argument(
        "--train_epochs", 
        type=int, 
        default=20,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda or cpu)"
    )
    parser.add_argument(
        "--skip_pretrain", 
        action="store_true",
        help="Skip pretraining and train from scratch"
    )
    parser.add_argument(
        "--pretrained_path", 
        type=str, 
        default=None,
        help="Path to pretrained model checkpoint (optional)"
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Configure logging to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"main_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Starting Medical VQA pipeline in {args.mode} mode")
        logger.info(f"Using device: {args.device}")
        
        # Prepare datasets with progress tracking
        if args.mode in ["prepare", "all"]:
            logger.info("Preparing datasets...")
            try:
                train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders(batch_size=args.batch_size)
                logger.info(f"Dataset preparation complete. Vocabulary size: {len(preprocessor.word2idx)}")
            except KeyboardInterrupt:
                logger.warning("Dataset preparation interrupted by user.")
                return
            except Exception as e:
                logger.error(f"Dataset preparation failed: {str(e)}")
                return
        
        # Continue with rest of the pipeline...
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user. Exiting gracefully...")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    
    # Pretrain model
    pretrained_path = args.pretrained_path
    if args.mode in ["pretrain", "all"] and not args.skip_pretrain:
        logger.info("Starting contrastive pretraining...")
        model, pretrained_path = pretrain_model(
            batch_size=args.batch_size,
            num_epochs=args.pretrain_epochs,
            device=args.device
        )
        plot_pretraining_metrics(log_file)
        logger.info(f"Pretraining complete. Model saved to {pretrained_path}")
    
    # Train model
    if args.mode in ["train", "all"]:
        logger.info("Starting model training...")
        if args.skip_pretrain:
            logger.info("Skipping pretraining, training from scratch")
            pretrained_path = None
        
        model, test_metrics = train_model(
            pretrained_path=pretrained_path,
            batch_size=args.batch_size,
            num_epochs=args.train_epochs,
            device=args.device
        )
        plot_training_metrics(log_file)
        logger.info(f"Training complete. Test metrics: {test_metrics}")
    
    logger.info("Pipeline execution completed successfully")

if __name__ == "__main__":
    main()