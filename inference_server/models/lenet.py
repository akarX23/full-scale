"""
LeNet AIPC model configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from .base import BaseModel_AIPC


class LeNet(nn.Module):
    """LeNet-5 style architecture for 32×32 grayscale input (10 classes)."""

    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 6, 5, stride=1, padding=0, bias=True)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(6, 16, 5, stride=1, padding=0, bias=True)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(16 * 5 * 5, 120, bias=True)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        relu    = nn.ReLU()
        x = relu(self.pool1(self.conv1(x)))
        x = relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        return self.linear3(x)


class LeNet_AIPC(BaseModel_AIPC):
    def __init__(self, batch_size: int = 1, torch_model_path: str = None, ov_model_path: str = None, bytes_per_element: int = 4):
        self.init_model = LeNet()
        self.example_input = torch.randn(1, 1, 32, 32)
        self.batch_size = batch_size
        super().__init__(torch_model_path, ov_model_path, bytes_per_element=bytes_per_element)

    def preprocess_raw_images(self, raw_images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        preprocessed_images = [transform(img) for img in raw_images]
        return torch.stack(preprocessed_images).numpy()

    def load_train_val_datasets(self, batch_size: int = 64, data_dir: str = "./data", max_samples=None):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        if max_samples and len(train_dataset) > max_samples:
            train_dataset = torch.utils.data.Subset(train_dataset, range(max_samples))
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

    def setup_training(self, learning_rate: float = 0.001, momentum: float = 0.9, device: str = "CPU", model=None):
        if model is not None:
            self.model = model
        self.train_device = device
        self.configure_cpu_training_threads(self.train_device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
