"""
BaseModel_AIPC and the analytics extraction helper.
"""

import copy
import math
import os

import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.optim as optim
import torchao

torch.serialization.add_safe_globals(
    [torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor]
)


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #

def extract_model_analytics(model: ov.Model, core: ov.Core, bytes_per_element: int = 4) -> list:
    graph_nodes = model.get_ordered_ops()

    layer_metrics = []

    npu_support = core.query_model(model, "NPU")
    gpu_support = core.query_model(model, "GPU")
    cpu_support = core.query_model(model, "CPU")

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
            curr_mac = (
                input_shape[0]
                * (kernel_shape[1] * kernel_shape[2] * kernel_shape[3])
                * (output_shape[2] * output_shape[3] * output_shape[1])
            )
            memory_traffic = (
                math.prod(input_shape) + math.prod(kernel_shape) + math.prod(output_shape)
            ) * bytes_per_element

        elif node_type == "MatMul":
            input_shape = node.input(0).get_shape()
            weights_shape = node.input(1).get_shape()
            output_shape = node.output(0).get_shape()
            curr_mac = input_shape[0] * input_shape[1] * output_shape[1]
            memory_traffic = (
                math.prod(input_shape) + math.prod(weights_shape) + math.prod(output_shape)
            ) * bytes_per_element

        arith_intensity = curr_mac / memory_traffic if memory_traffic != 0 else 0

        layer_metrics.append({
            "name": node.get_friendly_name(),
            "type": node_type,
            "macs": curr_mac,
            "memory_traffic": memory_traffic,
            "arith_intensity": arith_intensity,
            "supported_on": (
                ("NPU " if can_run_npu else "")
                + ("GPU " if can_run_gpu else "")
                + ("CPU" if can_run_cpu else "")
            ),
        })

    return layer_metrics


# --------------------------------------------------------------------------- #
# Base AIPC class
# --------------------------------------------------------------------------- #

class BaseModel_AIPC:
    # Set by child classes before super().__init__()
    model: nn.Module
    example_input: torch.Tensor
    batch_size: int
    init_model: nn.Module

    def __init__(self, torch_model_path: str = None, ov_model_path: str = None, bytes_per_element: int = 4):
        self.core: ov.Core = ov.Core()
        self.ov_model: ov.Model = None
        self.compiled_model: ov.CompiledModel = None
        self.infer_request: ov.InferRequest = None
        self.infer_requests: list[ov.InferRequest] = []
        self.optimal_nireq: int = 0
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

        if has_ov:
            self.ov_model = self.core.read_model(ov_model_path)
            self.ov_model.reshape([self.batch_size, *self.example_input.shape[1:]])
            bin_path = ov_model_path.replace(".xml", ".bin")
            if os.path.isfile(bin_path):
                self.model_file_size_bytes = os.path.getsize(bin_path)
            print(f"Loaded OV model from {ov_model_path}")

        if has_torch:
            self.init_model.load_state_dict(torch.load(torch_model_path, weights_only=True))
            self.init_model.eval()
            self.model_file_size_bytes = os.path.getsize(torch_model_path)
            print(f"Loaded torch model from {torch_model_path}, type: {type(self.init_model)}")
            self.restore_model_from_init_model()
        elif not has_ov:
            self.init_model.eval()
            self.restore_model_from_init_model()
            print("No valid model paths provided; using initialized in-memory torch model.")

        if self.ov_model is None:
            self.ov_model = ov.convert_model(self.model, example_input=self.example_input)
            self.ov_model.reshape([self.batch_size, *self.example_input.shape[1:]])

        self.analytics = extract_model_analytics(self.ov_model, self.core, bytes_per_element=self.bytes_per_element)

    def refresh_init_model(self, model: nn.Module = None, device=None):
        source_model = model if model is not None else self.model
        self.init_model = copy.deepcopy(source_model)
        if device is not None:
            self.init_model.to(device)
        self.init_model.eval()

    def restore_model_from_init_model(self, device=None) -> nn.Module:
        self.model = copy.deepcopy(self.init_model)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        return self.model

    def init_model_infer_object(self, device: str = "CPU"):
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        if device.upper() == "NPU":
            config.update({
                "NPU_TURBO": True,
                "NPU_COMPILATION_MODE_PARAMS": "optimization-level=2 performance-hint-override=latency",
            })
        from ..utils.constants import ADD_OPTIMAL_REQS
        self.compiled_model = self.core.compile_model(self.ov_model, device, config)
        # Create a pool of infer requests sized to fully saturate the device
        self.optimal_nireq = self.compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS") + ADD_OPTIMAL_REQS
        self.infer_requests = [self.compiled_model.create_infer_request() for _ in range(self.optimal_nireq)]
        self.infer_request = self.infer_requests[0]

    def predict_batch(self, preprocessed_images):
        infer_req = self.infer_request
        np.copyto(infer_req.get_input_tensor().data, preprocessed_images)
        infer_req.start_async()
        infer_req.wait()
        return infer_req.get_output_tensor().data.copy()

    def finetune_epoch(self, model: nn.Module = None) -> float:
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
        return total_loss / len(self.train_dataloader)

    def evaluate_accuracy(self, model=None, device=None) -> float:
        correct = 0
        total = 0

        if model is not None:
            torch_device = device.lower() if device is not None else self.train_device
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                for images, labels in self.test_dataloader:
                    images, labels = images.to(torch_device), labels.to(torch_device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        else:
            # OV path: pipeline over the full request pool to keep the device saturated.
            if not self.infer_requests:
                ov_device = device.upper() if device is not None else "CPU"
                self.init_model_infer_object(ov_device)
            pipeline = []
            nireq = len(self.infer_requests)

            for i, (images, labels) in enumerate(self.test_dataloader):
                req = self.infer_requests[i % nireq]
                if len(pipeline) == nireq:
                    old_req, old_labels = pipeline.pop(0)
                    old_req.wait()
                    predicted = np.argmax(old_req.get_output_tensor().data, axis=1)
                    total += old_labels.size(0)
                    correct += (predicted == old_labels.numpy()).sum().item()
                np.copyto(req.get_input_tensor().data, images.numpy())
                req.start_async()
                pipeline.append((req, labels))

            # Drain remaining in-flight requests
            for req, labels in pipeline:
                req.wait()
                predicted = np.argmax(req.get_output_tensor().data, axis=1)
                total += labels.size(0)
                correct += (predicted == labels.numpy()).sum().item()

        return correct / total

    def save_torch_model(self, path: str = "./models", model: nn.Module = None):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pth"))

    def save_ov_model(self, path: str = "./models", model: nn.Module = None):
        os.makedirs(path, exist_ok=True)
        print("Converting Torch model to OpenVINO format...")
        ov_model = ov.convert_model(model, example_input=self.example_input.to(self.train_device))
        ov_model.reshape([self.batch_size, *self.example_input.shape[1:]])
        ov.save_model(ov_model, os.path.join(path, "model.xml"))

    def preprocess_raw_images(self, raw_images):
        raise NotImplementedError

    def load_train_val_datasets(self, batch_size: int = 64, data_dir: str = "./data", max_samples=None):
        raise NotImplementedError

    def setup_training(self, learning_rate: float = 0.001, momentum: float = 0.9, device: str = "CPU", model=None):
        raise NotImplementedError
