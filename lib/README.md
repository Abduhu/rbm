# VMC Library

Custom utilities for Variational Monte Carlo simulations using jVMC.

## Structure

```
lib/
└── vmc/
    ├── __init__.py
    ├── hamiltonian.py     # Hamiltonian construction utilities
    ├── rbm.py             # Custom RBM architectures
    └── connectivity.py    # Connectivity pattern generators
```

## Installation

Add the `lib` directory to your Python path:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
```

Or in Jupyter notebooks:
```python
import sys
sys.path.append('./lib')
```

## Usage

### Building Hamiltonians from Pauli Strings

```python
from vmc.hamiltonian import build_hamiltonian_from_pauli_strings

# Method 1: String format (weight, pauli_string, sites)
terms = [
    (-1.0, "ZZ", (0, 1)),     # Z_0 Z_1 with coefficient -1.0
    (-0.7, "X", (0,)),         # X_0 with coefficient -0.7
    (0.5, "XYZ", (0, 1, 2)),  # X_0 Y_1 Z_2 with coefficient 0.5
]

H = build_hamiltonian_from_pauli_strings(terms)

# Method 2: Dictionary format (weight, {site: operator})
terms = [
    (-1.0, {0: 'Z', 1: 'Z'}),        # Z_0 Z_1
    (0.5, {0: 'X', 1: 'Y', 2: 'Z'}), # X_0 Y_1 Z_2
]

H = build_hamiltonian_from_pauli_strings(terms)
```

### Pretty Printing

```python
from vmc.hamiltonian import print_hamiltonian_terms

terms = [
    (-1.0, "ZZ", (0, 1)),
    (-0.7, "X", (0,)),
]

print_hamiltonian_terms(terms)
```

Output:
```
Hamiltonian Terms:
==================================================
Term 1: -1.0000 * Z_0 ⊗ Z_1
Term 2: -0.7000 * X_0
==================================================
```

## Examples

See the `tests/` directory for complete working examples:
- `test_hamiltonian_examples.py`: Comprehensive examples including:
  - Transverse Field Ising Model
  - Heisenberg XXZ Model
  - Multi-body interactions
  - Random Hamiltonians
  - Mixed format usage

Run the examples:
```bash
cd tests
python test_hamiltonian_examples.py
```

## Features

- ✅ Arbitrary number of qubits/spins
- ✅ Arbitrary Pauli strings (1-body, 2-body, 3-body, ..., N-body)
- ✅ Two input formats: string and dictionary
- ✅ Support for identity operators ('I')
- ✅ Non-adjacent site interactions
- ✅ Comprehensive error checking
- ✅ Type validation and helpful error messages

## API Reference

### `build_hamiltonian_from_pauli_strings(pauli_terms)`

Build a jVMC Hamiltonian from Pauli string terms.

**Parameters:**
- `pauli_terms` (list): List of tuples in one of two formats:
  - `(weight, pauli_string, sites)`: String format
  - `(weight, pauli_dict)`: Dictionary format

**Returns:**
- `jVMC.operator.BranchFreeOperator`: The constructed Hamiltonian

**Raises:**
- `ValueError`: If invalid Pauli operators or mismatched lengths

### `print_hamiltonian_terms(pauli_terms)`

Pretty print Hamiltonian terms for verification.

**Parameters:**
- `pauli_terms` (list): Same format as `build_hamiltonian_from_pauli_strings`

## Requirements

- jVMC
- JAX
- NumPy

---

## Custom RBM Architectures

### Creating Sparse RBMs

```python
from vmc.rbm import create_sparse_rbm
from vmc.connectivity import nearest_neighbor_pattern

# Define system
L = 8  # Number of spins
H = 4  # Number of hidden units

# Create connectivity pattern
connections = nearest_neighbor_pattern(L, H)

# Create sparse RBM
net = create_sparse_rbm(L, H, connections, bias=False)

# Wrap in jVMC NQS
import jVMC
psi = jVMC.vqs.NQS(net, seed=1234)

# Initialize (note: jVMC expects shape (batch, 1, L))
import jax.numpy as jnp
_ = psi(jnp.ones((1, 1, L)))
```

### Connectivity Pattern Generators

All patterns are in `vmc.connectivity` module:

**Basic Patterns:**
```python
from vmc.rbm import fully_connected_pattern

connections = fully_connected_pattern(L, H)
# Creates L×H connections (all-to-all)
```

**2. Local Connectivity** (each hidden unit connects to nearby visible units):
```python
from vmc.rbm import local_connectivity_pattern

connections = local_connectivity_pattern(L, H, receptive_field=3)
# Each hidden unit connects to 3 visible units in a local region
```

**3. Nearest-Neighbor** (minimal sparse connectivity):
```python
from vmc.rbm import nearest_neighbor_pattern

connections = nearest_neighbor_pattern(L, H)
# Each hidden unit connects to exactly 2 adjacent visible units
```

**4. Stripe Pattern** (non-overlapping groups):
```python
from vmc.rbm import stripe_pattern

connections = stripe_pattern(L, H, stripe_width=2)
# Divides visible units into stripes, each connected to one hidden unit
```

**5. Random Sparse** (random subset of connections):
```python
from vmc.rbm import random_sparse_pattern

connections = random_sparse_pattern(L, H, sparsity=0.3, seed=42)
# Randomly keeps 30% of all possible connections
```

**6. Custom Pattern**:
```python
# Define your own connectivity
custom_connections = [
    (0, 0), (1, 0),  # Visible 0,1 → Hidden 0
    (2, 1), (3, 1),  # Visible 2,3 → Hidden 1
    (0, 2), (4, 2),  # Visible 0,4 → Hidden 2
]
net = create_sparse_rbm(L, H, custom_connections)
```

### Visualizing Connectivity

```python
from vmc.rbm import visualize_connectivity
import matplotlib.pyplot as plt

connections = local_connectivity_pattern(8, 4, receptive_field=3)
fig, ax = visualize_connectivity(connections, 8, 4, title="Local Connectivity")
plt.show()
```

### Using in Ground State Search

```python
import jVMC
from jax import random
from vmc.rbm import create_sparse_rbm, local_connectivity_pattern

# Setup
L = 6
H = 3
g = -1.0

# Create sparse RBM
connections = local_connectivity_pattern(L, H, receptive_field=3)
net = create_sparse_rbm(L, H, connections, bias=False)
psi = jVMC.vqs.NQS(net, seed=5678)

# Build Hamiltonian (TFIM)
hamiltonian = jVMC.operator.BranchFreeOperator()
for i in range(L):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., 
        (jVMC.operator.Sz(i), jVMC.operator.Sz((i + 1) % L))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, 
        (jVMC.operator.Sx(i), )))

# Create sampler
sampler = jVMC.sampler.MCSampler(
    psi, (L,), random.PRNGKey(1234),
    numChains=20, sweepSteps=L, numSamples=300
)

# Optimize with TDVP
tdvp = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=1., diagonalShift=1e-2)
stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

for step in range(50):
    dp, _ = stepper.step(0, tdvp, psi.get_parameters(), 
                        hamiltonian=hamiltonian, psi=psi)
    psi.set_parameters(dp)
```

### RBM Features

- ✅ Custom sparse connectivity patterns
- ✅ Multiple pre-defined pattern generators
- ✅ Optional bias terms
- ✅ Support for complex and real dtypes
- ✅ Automatic parameter masking (no manual masking needed)
- ✅ Compatible with all jVMC samplers and optimizers
- ✅ Visualization utilities

### RBM Examples

See `tests/sparse_rbm_examples.py` for complete examples:
- Basic sparse RBM creation
- Ground state search with sparse RBM
- Comparing different connectivity patterns
- Custom connectivity patterns

Run the examples:
```bash
cd tests
python sparse_rbm_examples.py
```

### RBM Tests

Comprehensive unit tests in `tests/test_rbm.py`:
```bash
cd tests
python test_rbm.py
```

Tests include:
- Pattern generator validation
- Forward pass correctness
- Bias term handling
- Invalid connection detection
- Sparsity verification
- Different dtype support
