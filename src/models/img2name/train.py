
import time
import argparse
import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.img2name.img2name import Img2Name
from src.models.img2name.loaders import create_loader


def load_from_checkpoint(checkpoint_path, data_path, model_name, **kwargs):
    """
    Loads a model and dataset from a checkpoint.

    Args:
        checkpoint_path (str): Path to saved checkpoint.
        data_path (str): Path to dataset.
        model_name (str): Name of the model.
        **kwargs: Additional loader arguments.

    Returns:
        tuple: (model, dataset loader)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    with open(kwargs["maps_path"], "rb") as f:
        maps = pickle.load(f)
    with open(os.path.join(os.path.dirname(kwargs["maps_path"]), "init_params.pkl"), "rb") as f:
        init_params = pickle.load(f)

    model = Img2Name(maxlen=init_params['maxlen'], vocab_size=init_params['vocab_size']).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = create_loader(data_path, load_for=model_name, **kwargs)

    loader.dataset.vectorizer.load_maps(*maps)

    return model, loader


def train_model(model, loader, checkpoint_dir, epochs=1, learning_rate=0.001, save_interval=1, verbose=True):
    """
    Trains a model using the given dataset.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): PyTorch DataLoader for training data.
        checkpoint_dir (str): Directory to save checkpoints.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        save_interval (int): Checkpoint saving interval.
        verbose (bool): Whether to print detailed progress.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start_time = time.time()  # Track epoch time
        total_loss = 0.0
        batch_losses = []

        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True) if verbose else loader

        for batch_idx, ((images, sequences), targets) in enumerate(progress_bar):
            images, sequences, targets = images.to(device), sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())

            if batch_idx % 10 == 0 and verbose:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(loss=avg_loss)

        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - start_time  # Compute time per epoch

        # Show loss and time per epoch
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

        # Save checkpoints at intervals
        if epoch % save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, save_path)
            print(f"Checkpoint saved: {save_path}")

    final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, final_model_path)

    print(f"Training complete. Final model saved at {final_model_path}")


def train_from_scratch(arguments):
    """ Train a model from scratch. """

    loader = create_loader(
        data_path=arguments.data_path,
        load_for=arguments.model,
        img_col=arguments.img_col,
        name_col=arguments.name_col,
        img_dir=arguments.img_dir,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers
    )
    loader.dataset.vectorizer.save_maps(arguments.checkpoint_dir)
    vocab_size = loader.dataset.vectorizer.get_vocab_size()
    with open(os.path.join(arguments.checkpoint_dir, "init_params.pkl"), "wb") as f:
        pickle.dump({
            'maxlen': arguments.maxlen,
            'vocab_size': vocab_size
        }, f)
    model = Img2Name(maxlen=arguments.maxlen, vocab_size=vocab_size)
    train_model(model, loader, arguments.checkpoint_dir, arguments.epochs, arguments.learning_rate)


def train_from_checkpoint(arguments):
    """ Resume training from a checkpoint. """
    model, loader = load_from_checkpoint(
        checkpoint_path=arguments.checkpoint,
        data_path=arguments.data_path,
        model_name=arguments.model,
        img_dir=arguments.img_dir,
        batch_size=arguments.batch_size,
        maps_path=arguments.maps,
        name_col=arguments.name_col,
        img_col=arguments.img_col,
        maxlen=arguments.maxlen
    )
    train_model(model, loader, arguments.checkpoint_dir, arguments.epochs, arguments.learning_rate)


def parse_arguments():
    """ Parses command-line arguments for training. """
    parser = argparse.ArgumentParser(description="Train Img2Name model.")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for resuming training.")
    parser.add_argument("--maps", type=str, default=None, help="Path to character mappings file (.pkl).")

    parser.add_argument("--data_path", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing training images.")
    parser.add_argument("--img_col", type=str, required=True, help="Image column in dataset.")
    parser.add_argument("--name_col", type=str, required=True, help="Name column in dataset.")

    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--maxlen", type=int, default=3, help="Maximum sequence length.")

    return parser.parse_args()


def main(arguments):
    """ Main entry point for training. """

    if arguments.checkpoint:
        train_from_checkpoint(arguments)
    else:
        train_from_scratch(arguments)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
