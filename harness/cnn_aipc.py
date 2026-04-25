import copy
import openvino as ov
from openvino import Model
import torch
import math
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms, datasets
from datasets import load_dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import openvino.properties.hint as hints
import openvino.properties as props

_CPU_THREAD_TUNED = False


def _configure_cpu_training_threads(device: str) -> None:
    global _CPU_THREAD_TUNED
    if str(device).upper().split(":", 1)[0] != "CPU":
        return
    if _CPU_THREAD_TUNED:
        return

    cpu_cores = os.cpu_count() or 1
    torch.set_num_threads(cpu_cores)
    try:
        torch.set_num_interop_threads(cpu_cores)
    except RuntimeError:
        # Inter-op threads can only be configured once in some runtimes.
        pass
    _CPU_THREAD_TUNED = True

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
        relu = nn.ReLU()
        x = relu(self.pool1(self.conv1(x)))
        x = relu(self.pool2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = nn.Softmax(dim=1)(self.linear3(x))
        return x


def extract_model_analytics(model: ov.Model, core: ov.Core, bytes_per_element=4):
    graph_nodes = model.get_ordered_ops()
    
    layer_metrics = []

    available_devices = {device.split(".", 1)[0].upper() for device in core.available_devices}

    if "NPU" in available_devices:
        try:
            npu_support = core.query_model(model, "NPU")
        except Exception:
            npu_support = {}
    else:
        npu_support = {}

    if "GPU" in available_devices:
        try:
            gpu_support = core.query_model(model, "GPU")
        except Exception:
            gpu_support = {}
    else:
        gpu_support = {}

    if "CPU" in available_devices:
        try:
            cpu_support = core.query_model(model, "CPU")
        except Exception:
            cpu_support = {}
    else:
        cpu_support = {}

    for node in graph_nodes:
        
        curr_mac = 0
        memory_traffic = 0
        node_type = node.get_type_name()
        
        can_run_npu = npu_support.get(node.get_friendly_name(), False)
        can_run_gpu = gpu_support.get(node.get_friendly_name(), False)
        can_run_cpu = cpu_support.get(node.get_friendly_name(), False)
        
        if node_type == "Convolution":
            input_shape = node.input(0).get_shape()
            kernel_shape = node.input(1).get_shape()
            output_shape = node.output(0).get_shape()
            
            # Batch * (In_Channels * K_H * K_W) * (Out_H * Out_W * Out_Channels)
            curr_mac = input_shape[0] * (kernel_shape[1] * kernel_shape[2] * kernel_shape[3]) * (output_shape[2] * output_shape[3] * output_shape[1])
            memory_traffic = (math.prod(input_shape) + math.prod(kernel_shape) + math.prod(output_shape)) * bytes_per_element
        
        elif node_type == "MatMul":
            input_shape = node.input(0).get_shape()
            weights_shape = node.input(1).get_shape()
            output_shape = node.output(0).get_shape()
            
            # Batch * In_Features * Out_Features
            curr_mac = input_shape[0] * input_shape[1] * output_shape[1]
            memory_traffic = (math.prod(input_shape) + math.prod(weights_shape) + math.prod(output_shape)) * bytes_per_element
        
        arith_intensity = curr_mac / memory_traffic if memory_traffic != 0 else 0
        
        layer_metrics.append({
            "name": node.get_friendly_name(),
            "type": node_type,
            "macs": curr_mac,
            "memory_traffic": memory_traffic,
            "arith_intensity": arith_intensity,
            "supported_on": ("NPU " if can_run_npu else "") + ("GPU " if can_run_gpu else "") + ("CPU" if can_run_cpu else "")
        })
    
    return layer_metrics


def resolve_torch_device(device: str) -> str:
    """Map OpenVINO / generic device strings to a torch-compatible device name."""
    device_upper = str(device).upper().split(":", 1)[0]
    if device_upper == "GPU":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    if device_upper == "XPU":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
    return device_upper.lower()


def resolve_openvino_device(device: str, core: ov.Core) -> str:
    """Map torch-style device strings to an available OpenVINO hardware target."""
    device_upper = str(device).upper().split(":", 1)[0]
    if device_upper in {"CUDA", "XPU"}:
        device_upper = "GPU"

    available_devices = {d.split(".", 1)[0].upper() for d in core.available_devices}
    if device_upper in available_devices:
        return device_upper

    for fallback in ("CPU", "GPU", "NPU"):
        if fallback in available_devices:
            return fallback

    return device_upper

class BaseModel_AIPC:
    # Set by child classes before super().__init__()
    model: nn.Module
    example_input: torch.Tensor
    batch_size: int
    init_model: nn.Module

    def __init__(self, torch_model_path=None, ov_model_path=None, bytes_per_element=4):
        self.core: ov.Core = ov.Core()
        self.ov_model: ov.Model = None
        self.compiled_model: ov.CompiledModel = None
        self.infer_request: ov.InferRequest = None
        self.infer_requests: list[ov.InferRequest] = []
        self.analytics: list[dict] = []
        self.model_file_size_bytes: int = 0
        self.bytes_per_element: int = bytes_per_element
        self.train_dataloader: torch.utils.data.DataLoader = None
        self.test_dataloader: torch.utils.data.DataLoader = None
        self.train_device: torch.device = None
        self.learning_rate: float = None
        self.momentum: float = None
        self.criterion: nn.Module = None
        self.optimizer: optim.Optimizer = None

        has_ov = ov_model_path is not None and os.path.isfile(ov_model_path)
        has_torch = torch_model_path is not None and os.path.isfile(torch_model_path)

        # Load OV when provided.
        if has_ov:
            self.ov_model = self.core.read_model(ov_model_path)
            self.ov_model.reshape([self.batch_size, *self.example_input.shape[1:]])
            bin_path = ov_model_path.replace(".xml", ".bin")
            if os.path.isfile(bin_path):
                self.model_file_size_bytes = os.path.getsize(bin_path)
            print(f"Loaded OV model from {ov_model_path}")

        # Load torch weights when provided (independent of OV loading).
        if has_torch:
            self.init_model.load_state_dict(torch.load(torch_model_path, weights_only=True))
            self.init_model.eval()
            self.model_file_size_bytes = os.path.getsize(torch_model_path)
            print(f"Loaded torch model from {torch_model_path}, type: {type(self.init_model)}")
            self.restore_model_from_init_model()
        elif not has_ov:
            # No paths provided/valid: use the model already initialized by child class.
            self.init_model.eval()
            self.restore_model_from_init_model()
            print("No valid model paths provided; using initialized in-memory torch model.")

        # If OV is not available, convert from loaded torch model.
        if self.ov_model is None:
            self.ov_model = ov.convert_model(self.model, example_input=self.example_input)
            self.ov_model.reshape([self.batch_size, *self.example_input.shape[1:]])
        self.analytics = extract_model_analytics(self.ov_model, self.core, bytes_per_element=self.bytes_per_element)

    def refresh_init_model(self, model: torch.nn.Module=None, device=None):
        source_model = model if model is not None else self.model
        self.init_model = copy.deepcopy(source_model)
        if device is not None:
            self.init_model.to(device)
        self.init_model.eval()

    def restore_model_from_init_model(self, device=None):
        self.model = copy.deepcopy(self.init_model)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        return self.model
        
    def init_model_infer_object(self, device="CPU"):
        config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT}
        device_upper = resolve_openvino_device(device, self.core)
        if device_upper == "NPU":
            config.update({
                "NPU_TURBO": True,
                "NPU_COMPILATION_MODE_PARAMS": "optimization-level=2 performance-hint-override=latency",
            })
        # Wrap NPU/GPU with BATCH: to enable OpenVINO Automatic Batching; CPU is left unchanged
        compile_device = f"BATCH:{device_upper}" if device_upper in ("NPU", "GPU", "CPU") else device_upper
        self.compiled_model = self.core.compile_model(self.ov_model, compile_device, config)
        
        # Create a pool of infer requests sized to fully saturate the device
        optimal_nireq = self.compiled_model.get_property(props.optimal_number_of_infer_requests)
        self.infer_requests = [self.compiled_model.create_infer_request() for _ in range(optimal_nireq)]
        self.infer_request = self.infer_requests[0]
    
    def predict_batch(self, preprocessed_images):
        infer_req = self.infer_request
        # get_tensor idiom: write directly into the device-friendly input tensor (avoids extra copy)
        np.copyto(infer_req.get_input_tensor().data, preprocessed_images)
        # Async API: allows the device to pipeline work rather than block the calling thread
        infer_req.start_async()
        infer_req.wait()
        return infer_req.get_output_tensor().data.copy()
    
    def finetune_epoch(self, model: torch.nn.Module=None):
        total_loss = 0
        curr_model = model if model is not None else self.model
        curr_model.train()
        for images, labels in self.train_dataloader:
            images, labels = images.to(self.train_device), labels.to(self.train_device)
            self.optimizer.zero_grad()
            outputs = curr_model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate_accuracy(self, model=None, device=None):
        eval_device = device if device is not None else (self.train_device if self.train_device is not None else "CPU")
        correct = 0
        total = 0

        if model is not None:
            # PyTorch evaluation path
            _configure_cpu_training_threads(eval_device)
            torch_device = resolve_torch_device(device) if device is not None else self.train_device
            model.to(torch_device)
            model.eval()
            print("Evaluating with torch model on device:", torch_device)
            with torch.no_grad():
                for images, labels in self.test_dataloader:
                    images, labels = images.to(torch_device), labels.to(torch_device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        else:
            # OpenVINO evaluation path
            if self.compiled_model is None:
                self.init_model_infer_object(device=eval_device)
            print("Evaluating with OpenVINO model on device:", eval_device)
            for images, labels in self.test_dataloader:
                batch_np = images.numpy()
                outputs = self.predict_batch(batch_np)
                predicted = np.argmax(outputs, axis=1)
                total += labels.size(0)
                correct += (predicted == labels.numpy()).sum()

        return correct / total
    
    def save_torch_model(self, path="./models", model: torch.nn.Module=None):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pth"))
        
    def save_ov_model(self, path="./models", model: torch.nn.Module=None):
        os.makedirs(path, exist_ok=True)
        print("Converting Torch model to OpenVINO format...")
        model = ov.convert_model(model, example_input=self.example_input.to(self.train_device))
        model.reshape([self.batch_size, *self.example_input.shape[1:]])
        ov.save_model(model, os.path.join(path, "model.xml"))

    def preprocess_raw_images(self, raw_images):
        raise NotImplementedError

    def load_train_val_datasets(self, batch_size=64, data_dir="./data", max_samples=None):
        raise NotImplementedError

    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU", model=None):
        raise NotImplementedError

class LeNet_AIPC(BaseModel_AIPC):
    def __init__(self, batch_size=1, torch_model_path=None, ov_model_path=None, bytes_per_element=4):
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
    
    def load_train_val_datasets(self, batch_size=64, data_dir="./data", max_samples=None):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        if max_samples and len(train_dataset) > max_samples:
            train_dataset = torch.utils.data.Subset(train_dataset, range(max_samples))
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU", model=None):
        if model is not None:
            self.model = model
        self.train_device = device
        _configure_cpu_training_threads(self.train_device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
class AlexNet_AIPC(BaseModel_AIPC):
    def __init__(self, batch_size=1, torch_model_path=None, ov_model_path=None, bytes_per_element=4):
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
    
    def load_train_val_datasets(self, batch_size=64, data_dir="./data", max_samples=80000):
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
                image = self.transform(image)
                return image, label

        hf_dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=data_dir)

        train_split = hf_dataset["train"]
        if max_samples and len(train_split) > max_samples:
            train_split = train_split.select(range(max_samples))

        train_dataset = HFImageDataset(train_split, transform)
        test_dataset = HFImageDataset(hf_dataset["valid"], transform)

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU", model=None):
        if model is not None:
            self.model = model
        self.train_device = device
        _configure_cpu_training_threads(self.train_device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=momentum)

class VGG16_AIPC(BaseModel_AIPC):
    def __init__(self, batch_size=1, torch_model_path=None, ov_model_path=None, bytes_per_element=4):
        self.init_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
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

    def load_train_val_datasets(self, batch_size=64, data_dir="./data", max_samples=80000):
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
                image = self.transform(image)
                return image, label

        hf_dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=data_dir)

        train_split = hf_dataset["train"]
        if max_samples and len(train_split) > max_samples:
            train_split = train_split.select(range(max_samples))

        train_dataset = HFImageDataset(train_split, transform)
        test_dataset = HFImageDataset(hf_dataset["valid"], transform)

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU", model=None):
        if model is not None:
            self.model = model
        self.train_device = device
        _configure_cpu_training_threads(self.train_device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=momentum)

class ResNet18_AIPC(BaseModel_AIPC):
    def __init__(self, batch_size=1, torch_model_path=None, ov_model_path=None, bytes_per_element=4):
        self.init_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

    def load_train_val_datasets(self, batch_size=64, data_dir="./data", max_samples=60000):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        if max_samples and len(train_dataset) > max_samples:
            train_dataset = torch.utils.data.Subset(train_dataset, range(max_samples))
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    def setup_training(self, learning_rate=0.001, momentum=0.9, device="CPU", model=None):
        if model is not None:
            self.model = model
        self.train_device = device
        _configure_cpu_training_threads(self.train_device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model.to(self.train_device)
        self.criterion = nn.CrossEntropyLoss().to(self.train_device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=momentum)
