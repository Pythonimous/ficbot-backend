import pickle

import torch
import torch.nn as nn
import torchvision.models as models


class Img2Name(nn.Module):

    def __init__(self, maxlen, vocab_size, embedding_dim=128, fine_tune=True):
        """
        PyTorch version of Img2Name model with improvements.

        Args:
            maxlen (int): Maximum sequence length.
            vocab_size (int): Size of the vocabulary.
            fine_tune (bool): Whether to fine-tune MobileNetV3Large.
        """
        super(Img2Name, self).__init__()

        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # MobileNetV3 available to fine-tune if desired
        mobilenet = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
        self.transfer_out_dim = 960  # MobileNetV3Large output

        self.transfer = mobilenet.features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

        for param in self.transfer.parameters():
            param.requires_grad = fine_tune  # Fine-tune MobileNet if fine_tune=True

        # Bidirectional LSTM for Better Contextual Understanding
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=128, batch_first=True, bidirectional=True)

        # Layer Normalization and Dropout for Stability
        self.layer_norm = nn.LayerNorm(self.transfer_out_dim + 256)
        self.dropout = nn.Dropout(0.3)

        # Fusion -> Output layer
        self.fc1 = nn.Linear(self.transfer_out_dim + 256, 512)  # Fusion layer
        self.fc2 = nn.Linear(512, vocab_size)  # Output layer

    def forward(self, image_input, name_input):
        """
        Forward pass for Img2Name.

        Args:
            image_input (Tensor): Image tensor of shape (batch_size, 3, 224, 224).
            name_input (Tensor): Sequence tensor of shape (batch_size, maxlen).

        Returns:
            Tensor: Predicted character probabilities (batch_size, vocab_size).
        """
        # **🔹 Process Image Features**
        img_features = self.transfer(image_input)  # (batch_size, 960, 7, 7)
        img_features = self.global_avg_pool(img_features)  # (batch_size, 960, 1, 1)
        img_features = torch.flatten(img_features, start_dim=1)  # (batch_size, 960)

        # **🔹 Process Name Sequences**
        name_embedded = self.embedding(name_input)
        _, (name_features, _) = self.lstm(name_embedded)  # (2, batch_size, 128) due to bidirectional LSTM

        # Concatenate LSTM outputs from both directions
        name_features = torch.cat((name_features[0], name_features[1]), dim=1)  # (batch_size, 256)

        # **🔹 Fusion**
        x = torch.cat((img_features, name_features), dim=1)  # (batch_size, 960 + 256)
        x = self.layer_norm(x)  # Normalize before passing to dense layers
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        predictions = self.fc2(x)

        return predictions
    

    def image_embedding(self, image_input):
        """Extracts embeddings from an image input."""
        img_features = self.transfer(image_input)  # (batch_size, 960, 7, 7)
        img_features = self.global_avg_pool(img_features)  # (batch_size, 960, 1, 1)
        img_features = torch.flatten(img_features, start_dim=1)  # (batch_size, 960)
        return img_features  # Return the extracted embedding


    @classmethod
    def load_model(cls, weights_path, parameters_path):
        """
        Load a PyTorch model from a file.

        Args:
            model_path (str): Path to the model file.
            model_class (type): Model class to instantiate.
            init_params_path (str): Path to the file containing the model's initialization parameters.

        Returns:
            nn.Module: The loaded model.
        """

        with open(parameters_path, "rb") as f:
            init_params = pickle.load(f)

        model = cls(**init_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_state_dict = torch.load(weights_path, map_location=device)

        if 'model_state_dict' in model_state_dict:  # Checkpoint instead of just a state_dict
            model_state_dict = model_state_dict['model_state_dict']

        model.load_state_dict(model_state_dict)  # Load model for inference

        return model