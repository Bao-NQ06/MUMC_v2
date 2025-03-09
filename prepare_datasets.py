import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from data_preprocessing import VQAPreprocessor, VQADataset, create_dataloaders

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
RANDOM_SEED = 42
BASE_DIR = "e:/MUMC_v2"
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
VOCAB_PATH = os.path.join(CACHE_DIR, "vocab.pkl")

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def download_and_prepare_datasets():
    """
    Download PathVQA and VQA-RAD datasets from Hugging Face and prepare them for training.
    """
    logger.info("Downloading and preparing datasets...")
    
    # Download datasets from Hugging Face
    pathvqa_dataset = load_dataset("flaviagiammarino/path-vqa")
    vqa_rad_dataset = load_dataset("flaviagiammarino/vqa-rad")
    
    logger.info(f"PathVQA dataset splits: {pathvqa_dataset.keys()}")
    logger.info(f"VQA-RAD dataset splits: {vqa_rad_dataset.keys()}")
    
    # Convert to pandas DataFrames
    pathvqa_train_df = pd.DataFrame(pathvqa_dataset['train'])
    pathvqa_val_df = pd.DataFrame(pathvqa_dataset['validation']) if 'validation' in pathvqa_dataset else None
    pathvqa_test_df = pd.DataFrame(pathvqa_dataset['test']) if 'test' in pathvqa_dataset else None
    
    vqa_rad_df = pd.DataFrame(vqa_rad_dataset['train'])
    
    # Add dataset source column for tracking
    pathvqa_train_df['dataset_source'] = 'pathvqa'
    if pathvqa_val_df is not None:
        pathvqa_val_df['dataset_source'] = 'pathvqa'
    if pathvqa_test_df is not None:
        pathvqa_test_df['dataset_source'] = 'pathvqa'
    vqa_rad_df['dataset_source'] = 'vqa_rad'
    
    # Create validation split for VQA-RAD (since it doesn't have one)
    vqa_rad_train_df, vqa_rad_val_df = train_test_split(
        vqa_rad_df, test_size=0.1, random_state=RANDOM_SEED
    )
    
    # Create test split for VQA-RAD
    vqa_rad_train_df, vqa_rad_test_df = train_test_split(
        vqa_rad_train_df, test_size=0.1, random_state=RANDOM_SEED
    )
    
    # Combine datasets
    train_df = pd.concat([pathvqa_train_df, vqa_rad_train_df], ignore_index=True)
    val_df = pd.concat([pathvqa_val_df, vqa_rad_val_df], ignore_index=True) if pathvqa_val_df is not None else vqa_rad_val_df
    test_df = pd.concat([pathvqa_test_df, vqa_rad_test_df], ignore_index=True) if pathvqa_test_df is not None else vqa_rad_test_df
    
    # Save images to disk
    save_images(train_df, 'train')
    save_images(val_df, 'val')
    save_images(test_df, 'test')
    
    # Save processed DataFrames
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    
    # Analyze answer distribution
    analyze_answer_distribution(train_df)
    
    return train_df, val_df, test_df

def save_images(df, split_name):
    """
    Save images from dataset to disk.
    
    Args:
        df: DataFrame containing image data
        split_name: Name of the split (train, val, test)
    """
    logger.info(f"Saving {split_name} images to disk...")
    
    # Create directory for this split
    split_dir = os.path.join(IMAGE_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get image data
        image_data = row['image']
        
        # Generate a filename based on dataset source and index
        source = row['dataset_source']
        filename = f"{source}_{idx}.jpg"
        
        # Save image to disk
        image_path = os.path.join(split_dir, filename)
        
        # Update the DataFrame with the new image path (relative to IMAGE_DIR)
        df.at[idx, 'image'] = os.path.join(split_name, filename)
        
        # Save the image if it doesn't already exist
        if not os.path.exists(image_path):
            try:
                # Convert image data to PIL Image and save
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    # Handle case where image is stored as bytes
                    img = Image.open(io.BytesIO(image_data['bytes']))
                    img.save(image_path)
                elif isinstance(image_data, str) and os.path.exists(image_data):
                    # Handle case where image is a file path
                    img = Image.open(image_data)
                    img.save(image_path)
                else:
                    # Handle case where image is a numpy array or other format
                    img = Image.fromarray(image_data)
                    img.save(image_path)
            except Exception as e:
                logger.error(f"Error saving image {idx}: {str(e)}")

def analyze_answer_distribution(df):
    """
    Analyze the distribution of answers in the dataset.
    
    Args:
        df: DataFrame containing the dataset
    """
    logger.info("Analyzing answer distribution...")
    
    # Count answer frequencies
    answer_counts = df['answer'].value_counts()
    
    # Calculate yes/no question percentage
    yes_no_answers = df[df['answer'].str.lower().isin(['yes', 'no'])]
    yes_no_percentage = len(yes_no_answers) / len(df) * 100
    
    logger.info(f"Total unique answers: {len(answer_counts)}")
    logger.info(f"Top 10 most common answers: {answer_counts.head(10)}")
    logger.info(f"Yes/No questions: {yes_no_percentage:.2f}%")
    
    # Plot answer distribution
    plt.figure(figsize=(12, 6))
    answer_counts.head(20).plot(kind='bar')
    plt.title('Top 20 Answers Distribution')
    plt.xlabel('Answer')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'answer_distribution.png'))

def create_question_mask(question_tokens, pad_token_idx):
    """
    Create a boolean mask for question tokens where True indicates padding tokens.
    
    Args:
        question_tokens: Tensor of token indices of shape (B, L)
        pad_token_idx: Index of the padding token
        
    Returns:
        Boolean mask of shape (B, L) where True indicates padding tokens
    """
    return question_tokens == pad_token_idx

def prepare_dataloaders(batch_size=32, num_workers=4):
    """
    Prepare dataloaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, preprocessor)
    """
    # Check if processed data exists
    train_csv = os.path.join(DATA_DIR, "train.csv")
    val_csv = os.path.join(DATA_DIR, "val.csv")
    test_csv = os.path.join(DATA_DIR, "test.csv")
    
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        # Download and prepare datasets if not already done
        train_df, val_df, test_df = download_and_prepare_datasets()
    else:
        # Load existing processed data
        logger.info("Loading existing processed data...")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    
    # Initialize preprocessor
    preprocessor = VQAPreprocessor(
        image_dir=IMAGE_DIR,
        max_seq_len=77,
        img_size=224,
        min_token_freq=3,
        cache_dir=CACHE_DIR
    )
    
    # Check if vocabulary exists
    if os.path.exists(VOCAB_PATH):
        logger.info("Loading existing vocabulary...")
        preprocessor.load_vocabularies(VOCAB_PATH)
    else:
        logger.info("Building vocabulary...")
        preprocessor.build_vocabulary(
            questions=train_df["question"].tolist(),
            answers=train_df["answer"].tolist()
        )
        # Save vocabulary
        preprocessor.save_vocabularies(VOCAB_PATH)
    
    # Create datasets
    train_dataset = VQADataset(train_df, preprocessor, is_training=True)
    val_dataset = VQADataset(val_df, preprocessor, is_training=False)
    test_dataset = VQADataset(test_df, preprocessor, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, preprocessor

if __name__ == "__main__":
    # Prepare dataloaders
    train_loader, val_loader, test_loader, preprocessor = prepare_dataloaders()
    
    # Print dataset statistics
    logger.info(f"Vocabulary size: {len(preprocessor.word2idx)}")
    logger.info(f"Number of unique answers: {len(preprocessor.answer2idx)}")
    
    # Test dataloader
    for batch in train_loader:
        logger.info("Batch shapes:")
        logger.info(f"Images: {batch['images'].shape}")
        logger.info(f"Questions: {batch['question_tokens'].shape}")
        logger.info(f"Answer labels: {batch['answer_labels'].shape}")
        break