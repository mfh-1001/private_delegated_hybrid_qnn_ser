import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorV2


def strong_entangler_layer(n_qubits:int, reps:int):
    """
    Create a custom entangling ansatz layer with universal single-qubit gates and cyclic CNOTs.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        reps (int): Number of repeated entangling blocks.

    Returns:
        Tuple[QuantumCircuit, ParameterVector]: The quantum circuit and its parameter vector.
    """
    qc = QuantumCircuit(n_qubits)
    param_vector = ParameterVector('theta', 3 * n_qubits * reps)
    for i in range(reps):
        residual = 3*i*n_qubits
        for j in range(n_qubits):
            qc.u(param_vector[3*j+residual], param_vector[3*j+1+residual], param_vector[3*j+2+residual], qubit=j)
        assert n_qubits % 2 == 0, "Number of qubits must be even for this entanglement pattern."
        for q in range(0, n_qubits, 2):
            qc.cx(q, q+1)
        for q in range(1, n_qubits, 2):
            qc.cx(q, (q+1) % n_qubits)
        qc.barrier()
    return qc, param_vector

def angle_encoding_ry(feature_dimension: int) -> QuantumCircuit:
    """
    Create a feature map quantum circuit that encodes classical features into qubits using Ry rotations,
    as an alternative to the ZZFeatureMap.

    Args:
        feature_dimension (int): Number of features (and qubits) to encode.

    Returns:
        QuantumCircuit: Quantum circuit with Ry gates parameterized for feature encoding.
    """
    qc = QuantumCircuit(feature_dimension)
    x_params = ParameterVector("x", feature_dimension)
    for i in range(feature_dimension):
        qc.ry(x_params[i], i)
    qc.feature_params = x_params 
    return qc

def create_qnn(feature_dimension: int):
    """
    Create a Qiskit QNN (Quantum Neural Network) using a custom feature map and entangler ansatz.

    Args:
        feature_dimension (int): Number of features (and qubits) for the QNN model.

    Returns:
        EstimatorQNN: The constructed EstimatorQNN instance.
    """
    feature_map = angle_encoding_ry(feature_dimension)
    feature_map_decomp = feature_map.decompose()
    ansatz, _ = strong_entangler_layer(feature_dimension, 2)

    qc = QuantumCircuit(feature_dimension)
    qc.compose(feature_map_decomp, inplace=True)
    qc.compose(ansatz, inplace=True)
    estimator_prim = EstimatorV2()

    observables = [
        SparsePauliOp(['ZIII']),
        SparsePauliOp(['IZII']),
        SparsePauliOp(['IIZI']),
        SparsePauliOp(['IIIZ'])
    ]
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map_decomp.parameters,
        observables=observables,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator_prim
    )
    return qnn

def private_qnn(weights: torch.Tensor):
    """
    Placeholder function for a privacy-preserving QNN implementation.
    
    Args:
        weights (torch.Tensor): The weights to use for the QNN circuit.
    """
    pass

class PrivateQNN(nn.Module):
    """
    Custom layer for a quantum neural network with fixed weights. A variant that not relies on the Qiskit Machine Learning library
    and instead builds the quantum circuit manually. It allows for more control on how the forward pass is performed.
    """
    def __init__(self, weights:torch.Tensor, feature_dimension:int):
        super().__init__()
        self.weights = weights

    def qnn_circuit(self, weights, feature_dimension):
        """
        Build a Qiskit QuantumCircuit with the given weights and feature dimension.

        Args:
            weights (torch.Tensor): Weights to assign to ansatz parameters.
            feature_dimension (int): Number of qubits/features.

        Returns:
            QuantumCircuit: Fully constructed parameterized quantum circuit.
        """
        feature_map = angle_encoding_ry(feature_dimension)
        ansatz, _ = strong_entangler_layer(feature_dimension, 2)
        params = list(ansatz.parameters)
        params_dict = dict(zip(params, weights.tolist()))
        ansatz.assign_parameters(params_dict, inplace=True)
        qc = QuantumCircuit(feature_dimension)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        return qc


    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the PrivateQNN module. For each input in a batch, applies quantum circuit execution
        with input features mapped to input parameters, and fixed weights mapped to the ansatz. Returns
        batch of expectation values (one per observable if self.use_observables else one output per input).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dimension).

        Returns:
            torch.Tensor: Tensor of results stacked over the input batch.
        """
        outputs = []
        for i in range(x.shape[0]):
            qnn = self.qnn_circuit(self.weights, 4)
            input_params = list(qnn.parameters)
            input_params_dict = dict(zip(input_params, x[i].tolist()))
            qnn.assign_parameters(input_params_dict, inplace=True)
            estimator = EstimatorV2()
            observables = [
                SparsePauliOp(['ZIII']),
                SparsePauliOp(['IZII']),
                SparsePauliOp(['IIZI']),
                SparsePauliOp(['IIIZ'])
            ]
            result_list = []
            for observable in observables:
                exp_value = estimator.run(qnn, observable).result().values
                result_list.append(np.asarray(exp_value))
            # print(result_list)
            result_np = np.array(result_list)
            result = torch.from_numpy(result_np)
            outputs.append(result)
        outputs = torch.stack([r.flatten().to(dtype=x.dtype) for r in outputs]).to(x.device)
        return outputs
