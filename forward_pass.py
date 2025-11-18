import torch
from torch import nn
import numpy as np
from utils.utils import create_qnn, PrivateQNN
from qiskit_machine_learning.connectors import TorchConnector
from pathlib import Path

BASE_DIR = Path(__file__).parent
NUM_CLASSES = 5

torch.manual_seed(10)
np.random.seed(42)

class CNN_QNN(nn.Module):
    """
    CNN-QNN hybrid model for classification tasks, implemented with PyTorch and Qiskit Machine Learning.

    Input shape: [B, 1, 64, 129]
    Output:     [B, num_classes]
    """
    def __init__(self, num_classes: int, device, for_summary: bool = False):
        super().__init__()
        
        # Activation
        self.activation = nn.ReLU()

        # Convolutional Blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)        # [B, 32, 32, 64]

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        # [B, 64, 16, 32]

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)        # [B, 128, 8, 16]

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)        # [B, 256, 4, 8]

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)        # [B, 256, 2, 4]

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))         # [B, 256, 1, 1]
        self.flatten = nn.Flatten()                                             # [B, 256]
        self.cnn_norm = nn.LayerNorm(256)                      # [B, 256]

        # Quantum Neural Network (QNN) interface
        self.qnn_adapter = nn.Linear(256, 4).to(device)           # [B, 4]
        self.qnn = TorchConnector(create_qnn(4)).to(device)              # [B, 4]
        self.layer_norm = nn.LayerNorm(4).to(device)                      # [B, 4]
        self.fc_qnn = nn.Linear(4, 64).to(device)                 # [B, 64]
        self.classifier = nn.Linear(64, num_classes).to(device)   # [B, num_classes]

    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = self.pool4(self.activation(self.bn4(self.conv4(x))))
        x = self.pool5(self.activation(self.bn5(self.conv5(x))))
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.cnn_norm(x)
        # Quantum-inspired classification head
        x = self.qnn_adapter(x)
        x = self.qnn(x)
        x = self.fc_qnn(self.layer_norm(x))
        x = self.classifier(x)
        return x

class PrivateCNN_QNN(CNN_QNN):

    def __init__(self, num_classes:int, device, for_summary=False, n_shots:int=int(1e5), test_mode: bool = False, verbose: bool = True):
        super().__init__(num_classes, device)
        self.n_shots = n_shots
        self.test_mode = test_mode
        self.verbose = verbose

    def replace_qnn(self):
        FEATURE_DIMENSION = 4
        N_QUBITS_VERIF = 4
        self.qnn = PrivateQNN(self.qnn.weight.detach().clone(), FEATURE_DIMENSION, N_QUBITS_VERIF, self.n_shots, self.test_mode, self.verbose)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_shots = int(1e3)
test_mode = True
verbose = True

model = PrivateCNN_QNN(num_classes=NUM_CLASSES, device=device, n_shots=n_shots, test_mode=test_mode, verbose=verbose)
checkpoint_path = "models/hybrid_model.pt"
loaded_state = torch.load(checkpoint_path, map_location=device)['model_state']
model.load_state_dict(loaded_state)

model.to(device)

with torch.no_grad():
    x = torch.randn(5, 1, 64, 129).to(device)
    model.replace_qnn()
    print('Model private run result:', model(x))