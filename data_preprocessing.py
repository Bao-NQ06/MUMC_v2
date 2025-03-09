import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union, Set
import logging
import pickle
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VQAPreprocessor:
    """
    Preprocessor for VQA datasets (PathVQA and VQA-RAD).
    Handles image and text preprocessing, vocabulary construction, and dataset preparation.
    """
    def __init__(
        self,
        image_dir: str,
        max_seq_len: int = 77,
        img_size: int = 224,
        min_token_freq: int = 3,
        special_tokens: List[str] = ["<PAD>", "<UNK>", "<START>", "<END>"],
        cache_dir: Optional[str] = None
    ):
        self.image_dir = image_dir
        self.max_seq_len = max_seq_len
        self.img_size = img_size
        self.min_token_freq = min_token_freq
        self.special_tokens = special_tokens
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(image_dir), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize vocabularies and token mappings
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.answer2idx: Dict[str, int] = {}
        self.idx2answer: Dict[int, str] = {}
        
        # Initialize image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor of shape (3, H, W)
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert CMYK or grayscale to RGB
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            elif image.mode == 'L':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        return text.lower().strip().split()

    def build_vocabulary(self, questions: List[str], answers: List[str]) -> None:
        """
        Build vocabularies for questions and answers.
        
        Args:
            questions: List of question texts
            answers: List of answer texts
        """
        # Count token frequencies
        token_counter = Counter()
        for question in questions:
            tokens = self.tokenize(question)
            token_counter.update(tokens)
        
        # Add special tokens
        for token in self.special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
        
        # Add frequent tokens to vocabulary
        for token, count in token_counter.items():
            if count >= self.min_token_freq and token not in self.word2idx:
                self.word2idx[token] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = token
        
        # Build answer vocabulary (treat as classification)
        unique_answers = sorted(set(answers))
        self.answer2idx = {ans: idx for idx, ans in enumerate(unique_answers)}
        self.idx2answer = {idx: ans for ans, idx in self.answer2idx.items()}
        
        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        logger.info(f"Number of unique answers: {len(self.answer2idx)}")

    def encode_text(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode text into token indices.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (defaults to self.max_seq_len)
            
        Returns:
            Tensor of token indices
        """
        max_length = max_length or self.max_seq_len
        tokens = self.tokenize(text)
        
        # Truncate if necessary
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Convert tokens to indices
        indices = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        
        # Pad sequence
        padding_length = max_length - len(indices)
        if padding_length > 0:
            indices.extend([self.word2idx["<PAD>"]] * padding_length)
        
        return torch.tensor(indices, dtype=torch.long)

    def save_vocabularies(self, path: str) -> None:
        """Save vocabularies to disk."""
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "answer2idx": self.answer2idx,
            "idx2answer": self.idx2answer
        }
        with open(path, "wb") as f:
            pickle.dump(vocab_data, f)

    def load_vocabularies(self, path: str) -> None:
        """Load vocabularies from disk."""
        with open(path, "rb") as f:
            vocab_data = pickle.load(f)
        self.word2idx = vocab_data["word2idx"]
        self.idx2word = vocab_data["idx2word"]
        self.answer2idx = vocab_data["answer2idx"]
        self.idx2answer = vocab_data["idx2answer"]


class VQADataset(Dataset):
    """
    Dataset class for VQA data.
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        preprocessor: VQAPreprocessor,
        is_training: bool = True
    ):
        self.data = data_df
        self.preprocessor = preprocessor
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Process image
        image_path = os.path.join(self.preprocessor.image_dir, row["image"])
        image = self.preprocessor.preprocess_image(image_path)
        
        # Process question
        question_tokens = self.preprocessor.encode_text(row["question"])
        
        # Create question mask (True for padding tokens)
        question_mask = create_question_mask(question_tokens, self.preprocessor.word2idx["<PAD>"])
        
        # Process answer
        answer_label = torch.tensor(
            self.preprocessor.answer2idx[row["answer"]],
            dtype=torch.long
        )
        
        return {
            "images": image,
            "question_tokens": question_tokens,
            "question_mask": question_mask,
            "answer_labels": answer_label,
            "question_texts": row["question"],
            "answer_texts": row["answer"]
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: VQAPreprocessor,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        preprocessor: VQAPreprocessor instance
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = VQADataset(train_df, preprocessor, is_training=True)
    val_dataset = VQADataset(val_df, preprocessor, is_training=False)
    
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
    
    return train_loader, val_loader


def create_question_mask(question_tokens: torch.Tensor, pad_token_idx: int) -> torch.Tensor:
    """
    Create a boolean mask for question tokens where True indicates padding tokens.
    
    Args:
        question_tokens: Tensor of token indices
        pad_token_idx: Index of the padding token
        
    Returns:
        Boolean mask where True indicates padding tokens
    """
    return question_tokens == pad_token_idx

