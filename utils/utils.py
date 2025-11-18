import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorV2
from qiskit.primitives import StatevectorSampler as SamplerV2


# Test QNN model with no privacy

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
        SparsePauliOp(['I'*qubit+'Z'+'I'*(feature_dimension-qubit-1)])
        for qubit in range(feature_dimension) ]

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map_decomp.parameters,
        observables=observables,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator_prim
    )
    return qnn

class QNN(nn.Module):
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
        Forward pass through the QNN module. For each input in a batch, applies quantum circuit execution
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
            SparsePauliOp(['I'*qubit+'Z'+'I'*(x.shape[1]-qubit-1)])
            for qubit in range(x.shape[1]) ]
        
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

# Private Model

def angle_encoding_ry_private(feature_dimension: int, n_qubits_verif: int) -> tuple[QuantumCircuit, list]:
    """
    Create a private feature map quantum circuit that encodes classical features into qubits using Ry rotations
    with privacy-preserving obfuscation through random qubit shuffling. It can be improved introducing the first layer of U gates.

    Args:
        feature_dimension (int): Number of features (and qubits) to encode.
        n_qubits_verif (int): Number of additional verification qubits for privacy protection.

    Returns:
        tuple[QuantumCircuit, list]: The quantum circuit with user and server registers, 
                                    and the shuffled qubit indices for privacy mapping.
    """
    total_qubits = feature_dimension + n_qubits_verif

    # Create quantum registers with descriptive names
    user_register = QuantumRegister(2, name='User')
    server_register = QuantumRegister(total_qubits, name='Server')
    qc = QuantumCircuit(user_register, server_register, name='PrivateEncoding')
    
    # Parameter vector for feature encoding
    x_params = ParameterVector("x", total_qubits)

    # Create a randomly shuffled list of qubit indices for privacy obfuscation
    server_circuit = torch.randperm(total_qubits).tolist() # {server_position: circuit_position}
    
    # Apply Ry rotations with privacy-preserving qubit swapping
    for i in range(0, total_qubits, 2):
        # First qubit of the pair
        qubit0 = server_circuit[i]
        qc.ry(x_params[qubit0], user_register[0])
        qc.swap(user_register[0], server_register[i])

        # Second qubit of the pair (if exists)
        if i + 1 < total_qubits:
            qubit1 = server_circuit[i + 1]
            qc.ry(x_params[qubit1], user_register[1])
            qc.swap(user_register[1], server_register[i + 1])
    
    return qc, server_circuit

def strong_entangler_layer_private(n_qubits: int, n_qubits_verif: int, reps: int, server_circuit: list):
    """
    Create a custom entangling ansatz layer with universal single-qubit gates and cyclic CNOTs
    for privacy-preserving quantum circuits.

    Args:
        n_qubits (int): Number of primary qubits in the circuit.
        n_qubits_verif (int): Number of verification qubits for privacy protection.
        reps (int): Number of repeated entangling blocks.
        server_circuit (list): Shuffled qubit indices for privacy mapping.

    Returns:
        Tuple[QuantumCircuit, ParameterVector, dict]: The quantum circuit, parameter vector,
                                                     and updated circuit-to-server mapping.
    """
    # Constants for circuit structure
    R_DEMANDS_QUBIT = n_qubits // 3 + 1
    N_ANGLES = 3
    total_qubits = n_qubits + n_qubits_verif
    
    # Create quantum registers
    user_register = QuantumRegister(2, name='User')
    server_register = QuantumRegister(total_qubits, name='Server')
    qc = QuantumCircuit(user_register, server_register, name='PrivateEntangler')
    
    # Parameter vector for all rotation gates across all layers
    param_vector = ParameterVector('theta', N_ANGLES * total_qubits * reps)
    
    # Pre-compute qubit pairs for efficient entanglement pattern
    even_qubits = list(range(0, total_qubits, 2))
    odd_qubits = list(range(1, total_qubits, 2))
    
    # Create reverse mapping: circuit position -> server position
    circuit_server = {circuit_pos: server_pos for server_pos, circuit_pos in enumerate(server_circuit)}

    for rep in range(reps):
        param_offset = total_qubits * N_ANGLES * (rep+1)
        
        # Process even qubits with random entanglement patterns
        np.random.shuffle(even_qubits)
        for qubit0 in even_qubits:
            # Select random qubits for entanglement, ensuring adjacent qubit is included
            available_qubits = [q for q in range(total_qubits) if q != qubit0]
            np.random.shuffle(available_qubits)
            selected_qubits = available_qubits[:R_DEMANDS_QUBIT]

            # Ensure adjacent qubit is always included for nearest-neighbor connectivity
            adjacent_qubit = qubit0 + 1
            if adjacent_qubit not in selected_qubits:
                selected_qubits = selected_qubits[:-1] + [adjacent_qubit]
                np.random.shuffle(selected_qubits)

            for qubit1 in selected_qubits:
                # Randomly assign user register positions for privacy
                order0 = torch.randint(0, 2, (1,)).item()
                order1 = (order0 + 1) % 2
                
                # Get server positions for the qubits
                server_pos0 = circuit_server[qubit0]
                server_pos1 = circuit_server[qubit1]
                
                # Swap qubits to user registers for processing
                qc.swap(server_register[server_pos0], user_register[order0])
                qc.swap(server_register[server_pos1], user_register[order1])
                
                # Apply gates for adjacent qubits only
                if qubit1 == adjacent_qubit:
                    # Apply initial rotation gates only in first repetition
                    if rep == 0:
                        qc.u(param_vector[N_ANGLES*qubit0], param_vector[N_ANGLES*qubit0+1], 
                             param_vector[N_ANGLES*qubit0+2], user_register[order0])
                        qc.u(param_vector[N_ANGLES*qubit1], param_vector[N_ANGLES*qubit1+1], 
                             param_vector[N_ANGLES*qubit1+2], user_register[order1])
                    
                    # Apply entangling gate
                    qc.cx(user_register[order0], user_register[order1])

                # Randomly swap qubit positions back to server for privacy protection
                swap_positions = torch.randint(0, 2, (1,)).item() == 1
                if swap_positions:
                    # Swap positions: qubit0 goes to server_pos1, qubit1 goes to server_pos0
                    qc.swap(server_register[server_pos1], user_register[order0])
                    qc.swap(server_register[server_pos0], user_register[order1])
                    # Update circuit-to-server mapping
                    circuit_server[qubit0] = server_pos1
                    circuit_server[qubit1] = server_pos0
                else:
                    # Keep original positions
                    qc.swap(server_register[server_pos0], user_register[order0])
                    qc.swap(server_register[server_pos1], user_register[order1])

        # Process odd qubits with cyclic entanglement
        np.random.shuffle(odd_qubits)
        for qubit0 in odd_qubits:
            # Select random qubits for entanglement, ensuring cyclic target is included
            available_qubits = [q for q in range(total_qubits) if q != qubit0]
            np.random.shuffle(available_qubits)
            selected_qubits = available_qubits[:R_DEMANDS_QUBIT]

            # Determine cyclic target qubit based on position
            if qubit0 == n_qubits - 1:
                cyclic_target = 0  # Wrap around to first main qubit
            elif qubit0 == total_qubits - 1:
                cyclic_target = n_qubits  # Wrap around to first verification qubit
            else:
                cyclic_target = qubit0 + 1  # Next qubit in sequence

            # Ensure cyclic target is always included for proper entanglement
            if cyclic_target not in selected_qubits:
                selected_qubits = selected_qubits[:-1] + [cyclic_target]
                np.random.shuffle(selected_qubits)

            for qubit1 in selected_qubits:
                # Randomly assign user register positions for privacy
                order0 = torch.randint(0, 2, (1,)).item()
                order1 = (order0 + 1) % 2
                
                # Get current server positions for the qubits
                server_pos0 = circuit_server[qubit0]
                server_pos1 = circuit_server[qubit1]
                
                # Swap qubits to user registers for processing
                qc.swap(server_register[server_pos0], user_register[order0])
                qc.swap(server_register[server_pos1], user_register[order1])
                
                # Apply gates only for the cyclic target qubit
                if qubit1 == cyclic_target:
                    # Apply entangling gate
                    qc.cx(user_register[order0], user_register[order1])
                    
                    # Apply rotation gates except in final repetition to avoid over-parameterization
                    if rep != reps - 1:
                        qc.u(param_vector[N_ANGLES*qubit0 + param_offset], 
                             param_vector[N_ANGLES*qubit0 + 1 + param_offset], 
                             param_vector[N_ANGLES*qubit0 + 2 + param_offset], 
                             user_register[order0])
                        qc.u(param_vector[N_ANGLES*qubit1 + param_offset], 
                             param_vector[N_ANGLES*qubit1 + 1 + param_offset], 
                             param_vector[N_ANGLES*qubit1 + 2 + param_offset], 
                             user_register[order1])

                # Randomly swap qubit positions back to server for privacy protection
                swap_positions = torch.randint(0, 2, (1,)).item() == 1
                if swap_positions:
                    # Swap positions: qubit0 goes to server_pos1, qubit1 goes to server_pos0
                    qc.swap(server_register[server_pos1], user_register[order0])
                    qc.swap(server_register[server_pos0], user_register[order1])
                    # Update circuit-to-server mapping
                    circuit_server[qubit0] = server_pos1
                    circuit_server[qubit1] = server_pos0
                else:
                    # Keep original positions
                    qc.swap(server_register[server_pos0], user_register[order0])
                    qc.swap(server_register[server_pos1], user_register[order1])
        # Add barrier between repetitions for clarity
        qc.barrier()

    return qc, param_vector, circuit_server



def create_qnn_private(feature_dimension: int, n_qubits_verif: int):
    """
    Create a privacy-preserving Qiskit QNN (Quantum Neural Network) with verification capabilities.
    
    This function constructs a quantum neural network that incorporates privacy-preserving
    mechanisms through the use of verification qubits and secure quantum circuit design.
    The resulting QNN can be used for secure quantum machine learning applications.

    Args:
        feature_dimension (int): Number of input features (and main qubits) for the QNN model.
                                Must be even for the entanglement pattern to work correctly.
        n_qubits_verif (int): Number of verification qubits used for privacy protection.
                             Must be even for the entanglement pattern to work correctly.

    Returns:
        EstimatorQNN: The constructed EstimatorQNN instance with privacy-preserving features.
        
    Raises:
        ValueError: If feature_dimension or n_qubits_verif is not even.
        AssertionError: If the entanglement pattern requirements are not met.
    """
    # Validate input parameters
    if feature_dimension % 2 != 0:
        raise ValueError(f"feature_dimension must be even for entanglement pattern, got {feature_dimension}")
    if n_qubits_verif % 2 != 0:
        raise ValueError(f"n_qubits_verif must be even for entanglement pattern, got {n_qubits_verif}")
    
    # Configuration constants
    N_LAYERS = 2
    total_qubits = feature_dimension + n_qubits_verif

    # Build privacy-preserving feature map and ansatz
    feature_map, server_circuit = angle_encoding_ry_private(feature_dimension, n_qubits_verif)
    feature_map_decomp = feature_map.decompose()
    ansatz, _, circuit_server = strong_entangler_layer_private(
        feature_dimension, n_qubits_verif, N_LAYERS, server_circuit
    )

    # Construct quantum registers for privacy architecture
    user_register = QuantumRegister(2, name='User')
    server_register = QuantumRegister(total_qubits, name='Server')
    qc = QuantumCircuit(user_register, server_register, name='PrivateQNN')
    
    # Compose the complete quantum circuit
    qc.compose(feature_map_decomp, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    # Initialize quantum estimator primitive
    estimator_prim = EstimatorV2()

    # Create Z-observable for each qubit (main + verification)
    observables = [
        SparsePauliOp(['I' * qubit + 'Z' + 'I' * (total_qubits - qubit - 1)])
        for qubit in range(total_qubits)
    ]
        
    # Build the EstimatorQNN with proper parameter conversion
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=list(feature_map_decomp.parameters),  # Convert ParameterView to list
        observables=observables,
        weight_params=list(ansatz.parameters),  # Convert ParameterView to list
        input_gradients=True,
        estimator=estimator_prim
    )
    
    return qnn

class PrivateQNN(nn.Module):
    """
    Custom quantum neural network layer with privacy-preserving capabilities.
    
    This implementation provides manual control over quantum circuit execution without
    relying on the Qiskit Machine Learning library, enabling custom forward pass logic
    and verification mechanisms for secure quantum computations.
    
    The privacy-preserving mechanism works by:
    1. Shuffling qubit positions to hide the true circuit structure
    2. Using verification qubits to detect potential tampering
    3. Implementing secure multi-party computation protocols
    
    Args:
        weights (torch.Tensor): Pre-trained weights for the quantum circuit parameters.
                               Should match the expected parameter count for the ansatz.
        feature_dimension (int): Number of input features (must be even for entanglement pattern).
        n_qubits_verif (int): Number of verification qubits (must be even for entanglement pattern).
        n_shots (int): Number of quantum circuit shots for sampling. Higher values improve accuracy.
        test_mode (bool, optional): Enable additional testing and validation checks. Defaults to False.
        verbose (bool, optional): Enable verbose output for detecting attacks via the attack rate. Defaults to False.
        
    Raises:
        ValueError: If feature_dimension or n_qubits_verif is not even, or if n_shots is not positive.
        TypeError: If weights is not a torch.Tensor.
        
    Attributes:
        weights (torch.Tensor): Stored quantum circuit weights
        feature_dimension (int): Number of feature qubits
        n_qubits_verif (int): Number of verification qubits
        n_shots (int): Sampling shots for quantum execution
        test_mode (bool): Testing mode flag
        verbose (bool): Verbose output flag
        total_qubits (int): Total number of qubits (feature + verification)
        verifications (list): History of verification results for debugging
    """
    
    def __init__(self, weights: torch.Tensor, feature_dimension: int, n_qubits_verif: int, 
                 n_shots: int, test_mode: bool = False, verbose: bool = False):
        super().__init__()
        
        # Type validation
        if not isinstance(weights, torch.Tensor):
            raise TypeError(f"weights must be a torch.Tensor, got {type(weights)}")
        
        # Parameter validation with detailed error messages
        if feature_dimension % 2 != 0:
            raise ValueError(f"feature_dimension must be even for entanglement pattern, got {feature_dimension}")
        if n_qubits_verif % 2 != 0:
            raise ValueError(f"n_qubits_verif must be even for entanglement pattern, got {n_qubits_verif}")
        if n_shots <= 0:
            raise ValueError(f"n_shots must be positive, got {n_shots}")
        if feature_dimension <= 0:
            raise ValueError(f"feature_dimension must be positive, got {feature_dimension}")
        if n_qubits_verif <= 0:
            raise ValueError(f"n_qubits_verif must be positive, got {n_qubits_verif}")
        
        # Store configuration with proper tensor handling
        self.weights = weights.clone().detach().requires_grad_(False)
        self.feature_dimension = feature_dimension
        self.n_qubits_verif = n_qubits_verif
        self.n_shots = n_shots
        self.test_mode = test_mode
        self.verbose = verbose
        
        # Initialize verification tracking for debugging and security monitoring
        self.verifications = []
        
        # Cache derived values for efficiency
        self.total_qubits = feature_dimension + n_qubits_verif
        
        # Validate weights tensor dimensions
        expected_weight_count = self._calculate_expected_weights()
        if len(self.weights) != expected_weight_count:
            raise ValueError(f"Expected {expected_weight_count} weights for circuit configuration, "
                           f"got {len(self.weights)}")
    
    def _calculate_expected_weights(self) -> int:
        """Calculate the expected number of weights for the quantum circuit."""
        N_LAYERS = 2
        N_ANGLES = 3  # U gate has 3 parameters
        return self.feature_dimension * N_ANGLES * N_LAYERS
        
    def qnn_verifier_circuit(self, weights: torch.Tensor, n_qubits_verif: int) -> QuantumCircuit:
        """
        Build a verification quantum circuit for privacy-preserving quantum neural network.
        
        This method constructs a parameterized quantum circuit specifically designed for
        verification purposes in the private QNN implementation. The circuit combines
        angle encoding for feature mapping with a strongly entangled ansatz.

        Args:
            weights (torch.Tensor): Weights to assign to ansatz parameters. Must have
                                  the correct number of parameters for the ansatz.
            n_qubits_verif (int): Number of verification qubits to use in the circuit.

        Returns:
            QuantumCircuit: Fully parameterized quantum circuit ready for execution,
                          with feature map and ansatz layers composed.
                          
        Raises:
            ValueError: If weights tensor size doesn't match expected parameter count.
        """
        # Circuit configuration constants
        N_LAYERS = 2
        N_ANGLES = 3  # U gate has 3 parameters (theta, phi, lambda)
        
        # Validate weights tensor size
        expected_params = n_qubits_verif * N_ANGLES * N_LAYERS
        if len(weights) != expected_params:
            raise ValueError(
                f"Expected {expected_params} weights for {n_qubits_verif} qubits "
                f"and {N_LAYERS} layers, got {len(weights)}"
            )
        
        # Build circuit components
        feature_map = angle_encoding_ry(n_qubits_verif)
        ansatz, _ = strong_entangler_layer(n_qubits_verif, N_LAYERS)
        
        # Assign parameters to ansatz with proper error handling
        try:
            params = list(ansatz.parameters)
            params_dict = dict(zip(params, weights.detach().cpu().tolist()))
            ansatz.assign_parameters(params_dict, inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to assign parameters to ansatz: {e}")
        
        # Compose final circuit with proper naming
        qr = QuantumRegister(n_qubits_verif, name='qr')
        cr = ClassicalRegister(n_qubits_verif, name='cr')
        qc = QuantumCircuit(qr, cr, name='Verifier')
        
        # Add circuit layers
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # Add measurements (reversed order for proper bit ordering)
        qc.measure(qr, cr[::-1])
        
        return qc

    def qnn_private_circuit(self, weights: torch.Tensor, verification_weights: torch.Tensor,
                           feature_dimension: int, n_qubits_verif: int):
        """
        Build a privacy-preserving quantum circuit for the private QNN implementation.
        
        This method constructs a quantum circuit that combines user features with verification
        qubits to enable privacy-preserving quantum neural network computation. The circuit
        includes both feature encoding and parameterized ansatz layers.

        Args:
            weights (torch.Tensor): Main QNN weights for feature qubits ansatz parameters.
            verification_weights (torch.Tensor): Weights for verification qubits ansatz parameters.
            feature_dimension (int): Number of feature qubits (must be even).
            n_qubits_verif (int): Number of verification qubits for privacy protection.

        Returns:
            tuple: A tuple containing:
                - QuantumCircuit: Fully parameterized quantum circuit ready for execution
                - dict: Mapping from circuit positions to server register positions
                
        Raises:
            ValueError: If feature_dimension is not even or weight dimensions don't match.
        """
        # Circuit configuration constants
        N_LAYERS = 2
        N_ANGLES = 3  # U gate has 3 parameters (theta, phi, lambda)
        
        # Validate inputs
        if feature_dimension % 2 != 0:
            raise ValueError(f"Feature dimension must be even, got {feature_dimension}")
        
        total_qubits = feature_dimension + n_qubits_verif
        expected_main_params = N_ANGLES * feature_dimension * N_LAYERS
        expected_verif_params = N_ANGLES * n_qubits_verif * N_LAYERS
        
        if len(weights) != expected_main_params:
            raise ValueError(
                f"Expected {expected_main_params} main weights for {feature_dimension} qubits "
                f"and {N_LAYERS} layers, got {len(weights)}"
            )
        if len(verification_weights) != expected_verif_params:
            raise ValueError(
                f"Expected {expected_verif_params} verification weights for {n_qubits_verif} qubits "
                f"and {N_LAYERS} layers, got {len(verification_weights)}"
            )

        # Build circuit components with proper error handling
        try:
            feature_map, server_circuit = angle_encoding_ry_private(feature_dimension, n_qubits_verif)
            feature_map_decomp = feature_map.decompose()
            ansatz, _, circuit_server = strong_entangler_layer_private(
                feature_dimension, n_qubits_verif, N_LAYERS, server_circuit
            )
        except Exception as e:
            raise ValueError(f"Failed to build circuit components: {e}")

        # Prepare weight assignment with optimized memory allocation
        params = list(ansatz.parameters)
        weights_list = weights.detach().cpu().tolist()
        verification_weights_list = verification_weights.detach().cpu().tolist()
        
        # Pre-allocate total weights array for better performance
        total_params = N_LAYERS * total_qubits * N_ANGLES
        total_weights = [0.0] * total_params
        
        # Assign weights layer by layer with clear indexing
        for layer in range(N_LAYERS):
            layer_offset_total = layer * total_qubits * N_ANGLES
            layer_offset_main = layer * feature_dimension * N_ANGLES
            layer_offset_verif = layer * n_qubits_verif * N_ANGLES
            
            # Assign main feature weights (first feature_dimension qubits)
            for qubit in range(feature_dimension):
                qubit_offset_total = layer_offset_total + N_ANGLES * qubit
                qubit_offset_main = layer_offset_main + N_ANGLES * qubit
                for angle in range(N_ANGLES):
                    total_weights[qubit_offset_total + angle] = weights_list[qubit_offset_main + angle]
            
            # Assign verification weights (remaining n_qubits_verif qubits)
            for qubit in range(n_qubits_verif):
                qubit_offset_total = layer_offset_total + N_ANGLES * (qubit + feature_dimension)
                qubit_offset_verif = layer_offset_verif + N_ANGLES * qubit
                for angle in range(N_ANGLES):
                    total_weights[qubit_offset_total + angle] = verification_weights_list[qubit_offset_verif + angle]
        
        # Assign parameters to ansatz with error handling
        try:
            params_dict = dict(zip(params, total_weights))
            ansatz.assign_parameters(params_dict, inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to assign parameters to ansatz: {e}")

        # Construct final quantum circuit with proper naming
        user_register = QuantumRegister(2, name='User')
        server_register = QuantumRegister(total_qubits, name='Server')
        classical_register = ClassicalRegister(total_qubits, name='cr')
        qc = QuantumCircuit(user_register, server_register, classical_register, name='PrivateQNN')
        
        # Compose circuit components
        qc.compose(feature_map_decomp, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # Add measurements (reversed order for proper bit ordering)
        qc.measure(server_register, classical_register[::-1])

        return qc, circuit_server

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
        # Constants
        N_ANGLES = 3
        N_LAYERS = 2
        
        # Validate input
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Feature dimension must be even, got {x.shape[1]}")
        
        batch_size, feature_dim = x.shape
        outputs = []
        verifications = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Generate random verification weights and inputs for privacy protection
            verification_weights = torch.rand(N_ANGLES * N_LAYERS * self.n_qubits_verif) * 2 * torch.pi
            verif_random_list = (torch.rand(self.n_qubits_verif) * 2 * torch.pi).tolist()

            # Execute quantum circuit multiple times for statistical sampling
            total_shots_dict = self._execute_quantum_shots(
                x[i], verification_weights, verif_random_list, feature_dim
            )
            
            # Calculate expectation values for main qubits
            main_results = self._calculate_expectation_values(
                total_shots_dict, range(feature_dim)
            )
            outputs.append(torch.tensor(main_results, dtype=x.dtype))
            
            # Calculate expectation values for verification qubits
            verification_results = self._calculate_expectation_values(
                total_shots_dict, range(feature_dim, self.total_qubits)
            )
            
            # Run verification simulation for privacy validation
            simulation_results = self._run_verification_simulation(
                verification_weights, verif_random_list
            )
            
            # Compute verification RMSE to ensure privacy protection
            verification_rmse = self._compute_rmse(verification_results, simulation_results)
            verifications.append(verification_rmse)
            
            # Optional test mode for debugging and validation
            if self.test_mode:
                self._run_test_mode(x[i], main_results, feature_dim)
            if self.verbose:
                print(f'For data {i}, the attack rate is {verification_rmse}')
        
        # Stack outputs and store verifications for analysis
        outputs = torch.stack([r.flatten().to(dtype=x.dtype) for r in outputs]).to(x.device)
        self.verifications.append(verifications)
        
        return outputs
    
    def _execute_quantum_shots(self, input_sample, verification_weights, verif_random_list, feature_dim):
        """Execute quantum circuit for multiple shots and collect measurement results."""
        total_shots_dict = {}
        
        for shot in range(self.n_shots):
            # Create quantum circuit
            qnn, circuit_server = self.qnn_private_circuit(
                self.weights, verification_weights, feature_dim, self.n_qubits_verif
            )
            
            # Assign input parameters
            input_params = list(qnn.parameters)
            input_params_dict = dict(zip(input_params, input_sample.tolist() + verif_random_list))
            qnn.assign_parameters(input_params_dict, inplace=True)

            
            # Execute circuit
            sampler = SamplerV2()
            job = sampler.run([qnn], shots=1)
            counts = job.result()[0].data.cr.get_counts()
            
            # Optional circuit visualization for debugging
            # if shot % 100 == 0:
            #     fig = qnn.draw('mpl')
            #     fig.savefig(f'circuit_figures/circuit_shot_{shot}.png', dpi=300, bbox_inches='tight')
            
            # Process measurement result
            bitstring = list(counts.keys())[0]
            reordered_bitstring = self._reorder_bitstring(bitstring, circuit_server)
            
            # Optional debugging output
            # if shot % 10 == 0:
            #     print('Corr_dict:', circuit_server)
            #     print('Bitstring:  ', bitstring)
            #     print('Reorder_bit:', reordered_bitstring)
            
            # Accumulate results
            total_shots_dict[reordered_bitstring] = total_shots_dict.get(reordered_bitstring, 0) + 1
        
        return total_shots_dict
    
    def _reorder_bitstring(self, bitstring, circuit_server):
        """Reorder bitstring according to circuit_server mapping."""
        reordered_bits = ['0'] * len(bitstring)
        for circuit_position, server_position in circuit_server.items():
            reordered_bits[circuit_position] = bitstring[server_position]
        return ''.join(reordered_bits)
    
    def _calculate_expectation_values(self, shots_dict, qubit_indices):
        """Calculate Z-observable expectation values for specified qubits."""
        results = []
        
        for qubit_idx in qubit_indices:
            expectation_value = 0.0
            for bitstring, count in shots_dict.items():
                bit_value = int(bitstring[qubit_idx])
                z_eigenvalue = 1 - 2 * bit_value
                expectation_value += z_eigenvalue * count
            
            results.append(expectation_value / self.n_shots)
        
        return results
    
    def _run_verification_simulation(self, verification_weights, verif_random_list):
        """Run verification circuit simulation."""
        qnn = self.qnn_verifier_circuit(verification_weights, self.n_qubits_verif)
        
        # Assign parameters and execute
        input_params = list(qnn.parameters)
        input_params_dict = dict(zip(input_params, verif_random_list))
        qnn.assign_parameters(input_params_dict, inplace=True)
        
        sampler = SamplerV2()
        job = sampler.run([qnn], shots=self.n_shots)
        counts = job.result()[0].data.cr.get_counts()
        
        # Calculate expectation values
        return self._calculate_expectation_values(counts, range(self.n_qubits_verif))
    
    def _compute_rmse(self, results1, results2):
        """Compute Root Mean Square Error between two result sets.
        
        Args:
            results1: First set of results (list or array-like)
            results2: Second set of results (list or array-like)
            
        Returns:
            torch.Tensor: RMSE value as a scalar tensor
        """
        if len(results1) != len(results2):
            raise ValueError(f"Result sets must have same length: {len(results1)} vs {len(results2)}")
            
        tensor1 = torch.tensor(results1, dtype=torch.float32)
        tensor2 = torch.tensor(results2, dtype=torch.float32)
        mse = torch.mean((tensor1 - tensor2) ** 2)
        rmse = torch.sqrt(mse)
        # Normalize by the magnitude of the reference tensor to get relative error
        magnitude = torch.sqrt(torch.tensor(len(tensor2), dtype=torch.float32))
        normalized_rmse = rmse / (magnitude + 1e-8)  # Add small epsilon to avoid division by zero
        return normalized_rmse
    
    def _run_test_mode(self, input_sample, main_results, feature_dim):
        """Run test mode for debugging purposes."""
        qnn = self.qnn_verifier_circuit(self.weights, feature_dim)
        
        # Assign parameters and execute
        input_params = list(qnn.parameters)
        input_params_dict = dict(zip(input_params, input_sample.tolist()))
        qnn.assign_parameters(input_params_dict, inplace=True)
        
        sampler = SamplerV2()
        job = sampler.run([qnn], shots=self.n_shots)
        counts = job.result()[0].data.cr.get_counts()
        
        # Calculate test results and compare
        test_results = self._calculate_expectation_values(counts, range(feature_dim))
        print('Resultados test:', test_results)
        test_rmse = self._compute_rmse(test_results, main_results)
        
        print(f"Test RMSE: {test_rmse.item()}")

