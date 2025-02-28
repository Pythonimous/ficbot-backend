import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

from src.core.utils import get_image
from src.core.vectorizer import SequenceVectorizer


class ImageLoader(object):
    """Base class for image loading."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NameLoader(object):
    """Vectorizes names using a provided vectorizer."""
    
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def _get_sequences(self, name: str, maxlen: int, step: int = 1):
        """
        Converts a given name into multiple token index sequences for training.

        Each input name is split into overlapping sequences of length `maxlen`, 
        where each token is mapped to its corresponding index

        Args:
            name (str): The input character sequence.
            maxlen (int): Maximum sequence length.
            step (int): Step size for sequence generation.

        Returns:
            tuple:
                - vector_seq (numpy.ndarray): A 2D array of shape 
                `(num_sequences, maxlen)`, where each sequence contains 
                token indices
                - vector_next (numpy.ndarray): A 1D array of shape 
                `(num_sequences,)`, representing the next token for 
                each sequence as an integer index.
        """
        vector_sequences, vector_next = self.vectorizer.vectorize(
            name, maxlen=maxlen, step=step
        )
        
        return vector_sequences, vector_next


class ImageNameLoader(ImageLoader, NameLoader, Dataset):
    """PyTorch Dataset for Image + Name loading.

    Attributes:
        df (Pandas dataframe): Dataframe with a name column
        img_col (str): Label of a column with image names
        name_col (str): Label of a column with character names
        batch_size (int): Batch size (handled by DataLoader)
        img_dir (str): Path to a directory with images
        img_shape (tuple): Desired image shape for the pretrained algorithm
        transfer_net (str): Pretrained network for feature extraction
        vectorizer (SequenceVectorizer object): Preinitialized name vectorizer
        start_token (str): Token denoting the start of a name
        end_token (str): Token denoting the end of a name
        maxlen (int): Maximum sequence length
        step (int): Step size for sequence generation
    """

    def __init__(self, df, img_col, name_col, *,
                 img_dir: str,
                 image_shape: tuple = (224, 224, 3),
                 transfer_net: str = "mobilenet",
                 vectorizer=None,
                 start_token: str = "@",
                 end_token: str = "$",
                 maxlen: int = 3,
                 step: int = 1,
                 shuffle: bool = True):

        self.df = df.copy()
        self.img_col = img_col
        self.name_col = name_col
        self.img_dir = img_dir
        self.img_shape = image_shape
        self.transfer_net = transfer_net
        self.maxlen = maxlen
        self.step = step

        # Apply start/end tokens to names
        self.df[name_col] = self.df[self.name_col].map(lambda x: start_token * maxlen + x + end_token)

        # Initialize vectorizer
        if vectorizer is None:
            self.vectorizer = SequenceVectorizer(corpus=self.df[name_col].tolist(), ood_token="?")
        else:
            self.vectorizer = vectorizer

        super().__init__(self.vectorizer)

        # Shuffle dataset if requested
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns one sample (image, sequence) at a time.
        """
        row = self.df.iloc[index]
        name = row[self.name_col]
        img_path = os.path.join(self.img_dir, row[self.img_col])

        # Vectorize name to get sequences
        X_seq, y = self._get_sequences(name, maxlen=self.maxlen)

        # Convert to tensors
        X_img_tensor = get_image(img_path, self.img_shape[:2], self.transfer_net)  # (224, 224)
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return (X_img_tensor, X_seq_tensor), y_tensor


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and ensure proper
    alignment of images and sequences.

    Args:
        batch (list): A list of tuples ((X_img, X_seq), y) from Dataset.__getitem__

    Returns:
        tuple: 
            - X_img_batch (tensor): Shape (n_sequences, 3, 224, 224)
            - X_seq_batch (tensor): Padded sequences, shape (n_sequences, max_seq_len)
            - y_batch (tensor): Target token indices, shape (n_sequences,)
    """
    X_img_list, X_seq_list, y_list = [], [], []
    
    for (img, seq), y in batch:
        num_sequences = seq.shape[0]  # Number of sequences generated from this name

        # Expand image tensor for each sequence (same image, different text sequences)
        X_img_expanded = img.unsqueeze(0).expand(num_sequences, -1, -1, -1)  # (num_sequences, 3, 224, 224)
        X_img_list.append(X_img_expanded)

        X_seq_list.extend(seq)  # Append tokenized sequences
        y_list.extend(y.tolist())  # Append next token indices (already integers)

    # Convert images to a single batch tensor
    X_img_batch = torch.cat(X_img_list, dim=0)  # (n_sequences, 3, 224, 224)

    # Pad tokenized sequences to match max sequence length in batch
    X_seq_batch = pad_sequence(X_seq_list, batch_first=True, padding_value=0).long()  # (n_sequences, max_seq_len)

    y_batch = torch.tensor(y_list, dtype=torch.long)  # (n_sequences,)

    return (X_img_batch, X_seq_batch), y_batch


def create_loader(data_path, *, load_for, batch_size=1, shuffle=True, num_workers=4, **kwargs):
    """Creates a PyTorch DataLoader from the given dataset.

    Args:
        data_path (str): Path to the CSV file containing data.
        load_for (str): Name of the algorithm to load for.
        batch_size (int, optional): Number of samples per batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        num_workers (int, optional): Number of parallel workers for loading data. Defaults to 4.
        **kwargs: Additional keyword arguments passed to the dataset.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataframe = pd.read_csv(data_path)
    models = {"simple_img_name"}

    assert load_for in models, f"No such model.\nAvailable models: {', '.join(models)}."

    if load_for == "simple_img_name":
        loader = ImageNameLoader(
            dataframe, kwargs["img_col"], kwargs["name_col"],
            img_dir=kwargs["img_dir"],
            image_shape=kwargs.get("img_shape", (224, 224, 3)),
            transfer_net=kwargs.get("transfer_net", "mobilenet"),
            vectorizer=kwargs.get("vectorizer", None),
            start_token=kwargs.get("start_token", "@"),
            end_token=kwargs.get("end_token", "$"),
            maxlen=kwargs.get("maxlen", 3),
            step=kwargs.get("step", 1),
            shuffle=shuffle  # Shuffle is now handled by DataLoader, but we shuffle the DataFrame initially
        )

        return DataLoader(loader, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
