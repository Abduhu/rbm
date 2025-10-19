"""
Test examples for Hamiltonian builder

These examples demonstrate how to use the build_hamiltonian_from_pauli_strings
function to construct various quantum Hamiltonians.
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import jVMC
from vmc.hamiltonian import build_hamiltonian_from_pauli_strings, print_hamiltonian_terms


def test_transverse_field_ising_model():
    """
    Example 1: Transverse Field Ising Model
    
    Hamiltonian: H = -Σ_i Z_i Z_{i+1} + g Σ_i X_i
    """
    print("\n" + "="*70)
    print("Example 1: Transverse Field Ising Model")
    print("="*70)
    
    L = 10
    g = -0.7
    
    # Build using the function
    terms_tfim = []
    for l in range(L):
        terms_tfim.append((-1.0, "ZZ", (l, (l+1) % L)))  # ZZ interaction
        terms_tfim.append((g, "X", (l,)))                 # Transverse field
    
    print_hamiltonian_terms(terms_tfim[:6])  # Print first 6 terms
    H_tfim = build_hamiltonian_from_pauli_strings(terms_tfim)
    print(f"✓ Hamiltonian built with {len(terms_tfim)} terms")
    print(f"  System size: L = {L}")
    print(f"  Transverse field: g = {g}")
    
    return H_tfim


def test_heisenberg_model():
    """
    Example 2: Heisenberg XXZ Model
    
    Hamiltonian: H = J_xy Σ_i (X_i X_{i+1} + Y_i Y_{i+1}) + J_z Σ_i Z_i Z_{i+1}
    """
    print("\n" + "="*70)
    print("Example 2: Heisenberg XXZ Model")
    print("="*70)
    
    L = 6
    Jxy = 1.0   # XY coupling
    Jz = 1.5    # Z coupling
    
    terms_heisenberg = []
    for l in range(L):
        next_site = (l + 1) % L
        terms_heisenberg.append((Jxy, "XX", (l, next_site)))
        terms_heisenberg.append((Jxy, "YY", (l, next_site)))
        terms_heisenberg.append((Jz, "ZZ", (l, next_site)))
    
    print_hamiltonian_terms(terms_heisenberg[:6])
    H_heisenberg = build_hamiltonian_from_pauli_strings(terms_heisenberg)
    print(f"✓ Heisenberg Hamiltonian built with {len(terms_heisenberg)} terms")
    print(f"  System size: L = {L}")
    print(f"  J_xy = {Jxy}, J_z = {Jz}")
    
    return H_heisenberg


def test_custom_multibody():
    """
    Example 3: Custom Hamiltonian with multi-body interactions
    
    Demonstrates dictionary format and arbitrary multi-body terms.
    """
    print("\n" + "="*70)
    print("Example 3: Custom Multi-Body Hamiltonian (Dictionary Format)")
    print("="*70)
    
    terms_custom = [
        # Two-body terms
        (-1.0, {0: 'Z', 1: 'Z'}),
        (-1.0, {1: 'Z', 2: 'Z'}),
        
        # Three-body term
        (0.5, {0: 'X', 1: 'Y', 2: 'Z'}),
        
        # Four-body term
        (0.3, {0: 'Z', 1: 'Z', 2: 'Z', 3: 'Z'}),
        
        # Single-body terms
        (-0.7, {0: 'X'}),
        (-0.7, {1: 'X'}),
        
        # Non-sequential sites
        (0.2, {0: 'X', 3: 'X'}),
    ]
    
    print_hamiltonian_terms(terms_custom)
    H_custom = build_hamiltonian_from_pauli_strings(terms_custom)
    print(f"✓ Custom Hamiltonian built with {len(terms_custom)} terms")
    print("  Includes 2, 3, and 4-body interactions")
    
    return H_custom


def test_random_hamiltonian():
    """
    Example 4: Random Hamiltonian with arbitrary Pauli strings
    
    Generates random Pauli terms for testing.
    """
    print("\n" + "="*70)
    print("Example 4: Random Arbitrary Pauli Hamiltonian")
    print("="*70)
    
    import numpy as np
    
    np.random.seed(42)
    num_sites = 8
    num_terms = 10
    
    terms_random = []
    pauli_ops = ['X', 'Y', 'Z']
    
    for _ in range(num_terms):
        # Random weight
        weight = np.random.uniform(-1, 1)
        
        # Random number of sites involved (1 to 4 body interactions)
        num_bodies = np.random.randint(1, 5)
        
        # Random sites (no repetition)
        sites = tuple(np.random.choice(num_sites, size=num_bodies, replace=False))
        
        # Random Pauli string
        pauli_string = ''.join(np.random.choice(pauli_ops, size=num_bodies))
        
        terms_random.append((weight, pauli_string, sites))
    
    print_hamiltonian_terms(terms_random)
    H_random = build_hamiltonian_from_pauli_strings(terms_random)
    print(f"✓ Random Hamiltonian built with {len(terms_random)} terms")
    print(f"  System size: {num_sites} qubits")
    print(f"  Interaction range: 1 to 4 body terms")
    
    return H_random


def test_mixed_format():
    """
    Example 5: Mixed format usage
    
    Shows that you can mix string and dictionary formats.
    """
    print("\n" + "="*70)
    print("Example 5: Mixed Format Hamiltonian")
    print("="*70)
    
    terms_mixed = [
        # String format
        (-1.0, "ZZ", (0, 1)),
        (-1.0, "ZZ", (1, 2)),
        
        # Dictionary format
        ({0: 'X', 1: 'Y'}, 0.5),  # Note: order swapped accidentally - will fail
    ]
    
    # Actually, let's fix this - weight must come first
    terms_mixed = [
        # String format
        (-1.0, "ZZ", (0, 1)),
        (-1.0, "ZZ", (1, 2)),
        
        # Dictionary format
        (0.5, {0: 'X', 1: 'Y'}),
        
        # More string format
        (-0.3, "XYZ", (0, 1, 2)),
    ]
    
    print_hamiltonian_terms(terms_mixed)
    H_mixed = build_hamiltonian_from_pauli_strings(terms_mixed)
    print(f"✓ Mixed format Hamiltonian built with {len(terms_mixed)} terms")
    print("  Using both string and dictionary formats")
    
    return H_mixed


def run_all_examples():
    """Run all example tests"""
    print("\n" + "#"*70)
    print("# Hamiltonian Builder Examples")
    print("#"*70)
    
    try:
        H1 = test_transverse_field_ising_model()
        H2 = test_heisenberg_model()
        H3 = test_custom_multibody()
        H4 = test_random_hamiltonian()
        H5 = test_mixed_format()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70 + "\n")
        
        return {
            'tfim': H1,
            'heisenberg': H2,
            'custom': H3,
            'random': H4,
            'mixed': H5
        }
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run all examples when script is executed directly
    run_all_examples()
