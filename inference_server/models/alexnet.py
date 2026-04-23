"""
AlexNet AIPC model configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import AlexNet_Weights, alexnet

from .base import BaseModel_AIPC


class AlexNet_AIPC(BaseModel_AIPC):
    def __init__(self, batch_size: int = 1, torch_model_path: str = None, ov_model_path: str = None, bytes_per_element: int = 4):
        self.init_model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.example_input = torch.randn(1, 3, 224, 224)
        self.batch_size = batch_size
        super().__init__(torch_model_path, ov_model_path, bytes_per_element=bytes_per_element)

    def preprocess_raw_images(self, raw_images):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocessed_images = [transform(img) for img in raw_images]
        return torch.stack(preprocessed_images).numpy()

    def load_train_val_datasets(self, batch_size: int = 64, data_dir: str = "./data", max_samples: int = 80000):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        class HFImageDataset(torch.utils.data.Dataset):
            def __init__(self, hf_data, transform):
                self.hf_data = hf_data
                self.transform = transform

            def __len__(self):
                return len(self.hf_data)

            def __getitem__(self, idx):
                item = self.hf_data[idx]
                image = item["image"].convert("RGB")
                label = item["label"]
                return self.transform(image), label

        hf_dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=data_dir)
        train_split = hf_dataset["train"]
        if max_samples and len(train_split) > max_samples:
            train_split = train_split.select(range(max_samples))

        self.train_dataloader = torch.utils.data.DataLoader(
            HFImageDataset(train_split, transform), batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            HFImageDataset(hf_dataset["valid"], transform), batch_size=batch_size, shuffle=False, drop_last=True
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
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=momentum
        )
