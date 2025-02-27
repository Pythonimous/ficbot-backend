import unittest
import os
import torch
import pandas as pd

from src.core.loaders_torch import create_loader


class TestLoaders(unittest.TestCase):
    """Tests for PyTorch-based loaders"""

    def setUp(self):
        """Set up test environment, load dataset, and initialize loader."""
        self.current_dir = os.path.dirname(__file__)
        self.df_path = os.path.join(self.current_dir, "files/data/loaders/img_name.csv")
        self.img_dir = os.path.join(self.current_dir, "files/data/loaders/images")
        self.image_shape = (224, 224, 3)
        self.maxlen = 3
        self.batch_size = 5

        # Use create_loader instead of manually creating DataLoader
        self.data_loader = create_loader(
            data_path=self.df_path,
            load_for="simple_img_name",
            img_col="image",
            name_col="eng_name",
            img_dir=self.img_dir,
            batch_size=self.batch_size,
            img_shape=self.image_shape,
            maxlen=self.maxlen,
            shuffle=False
        )

    def test_image_name_loader(self):
        """Test that the ImageNameLoader correctly loads images and sequences."""
        torch.manual_seed(42)  # Ensure reproducibility

        for (X_img_batch, X_seq_batch), y_batch in self.data_loader:
            batch_names = pd.read_csv(self.df_path)["eng_name"][:self.batch_size].tolist()  # Get names in batch

            # Compute n_sequences dynamically
            n_sequences = sum(len(name) + 1 for name in batch_names)  # +1 for end token

            # Check tensor types
            self.assertIsInstance(X_img_batch, torch.Tensor)
            self.assertIsInstance(X_seq_batch, torch.Tensor)
            self.assertIsInstance(y_batch, torch.Tensor)

            # Expected image batch shape: (n_sequences, C, H, W)
            expected_img_batch_shape = (n_sequences, 3, 224, 224)  # PyTorch uses (C, H, W)
            self.assertEqual(X_img_batch.shape, expected_img_batch_shape)

            # Expected sequence batch shape: (n_sequences, maxlen)
            self.assertEqual(X_seq_batch.shape, (n_sequences, self.maxlen))

            # Expected output shape: (n_sequences, vocab_size)
            self.assertEqual(y_batch.shape, (n_sequences,))

            break  # Only test the first batch


if __name__ == "__main__":
    unittest.main()
