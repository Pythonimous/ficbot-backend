import unittest
import torch
import torch.nn as nn

from src.models.img2name.img2name_torch import Img2Name


class TestModels(unittest.TestCase):

    def test_compile(self):
        """Test if the Img2Name model initializes and passes a forward pass."""
        
        # Initialize model
        simple_name_torch = Img2Name(maxlen=3, vocab_size=420)

        # Ensure it's an instance of nn.Module (PyTorch model)
        self.assertIsInstance(simple_name_torch, nn.Module)

        sample_image = torch.randn(1, 3, 224, 224)  # (batch_size=1, C=3, H=224, W=224)
        sample_name = torch.randint(0, 420, (1, 3), dtype=torch.long)  # ✅ Token indices, not one-hot

        # Forward pass
        output = simple_name_torch(sample_image, sample_name)

        # Check output shape
        self.assertEqual(output.shape, (1, 420))  # (batch_size, vocab_size)


if __name__ == "__main__":
    unittest.main()