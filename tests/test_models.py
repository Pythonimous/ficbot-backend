import unittest
import torch
import torch.nn as nn

from src.models.img2name.img2name import Img2Name


class TestModels(unittest.TestCase):

    def test_compile_img2name(self):
        """Test if the Img2Name model initializes and passes a forward pass."""
        
        img2name = Img2Name(maxlen=3, vocab_size=420)
        self.assertIsInstance(img2name, nn.Module)

        sample_image = torch.randn(1, 3, 224, 224)  # (batch_size=1, C=3, H=224, W=224)
        sample_name = torch.randint(0, 420, (1, 3), dtype=torch.long)
        output = img2name(sample_image, sample_name)

        self.assertEqual(output.shape, (1, 420))  # (batch_size, vocab_size)


if __name__ == "__main__":
    unittest.main()