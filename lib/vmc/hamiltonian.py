"""
Hamiltonian Builder for jVMC

This module provides utilities to construct jVMC Hamiltonians from 
arbitrary Pauli strings and weights.
"""

import jVMC


def build_hamiltonian_from_pauli_strings(pauli_terms):
    """
    Build a jVMC Hamiltonian from a list of Pauli string terms.
    
    Parameters:
    -----------
    pauli_terms : list of tuples
        Each tuple is either:
        - (weight, pauli_string, sites) where:
            * weight: float, coefficient of the term
            * pauli_string: str, e.g., "XX", "ZZ", "XYZ", "I" (identity)
            * sites: tuple/list of int, site indices for each Pauli operator
        
        OR
        
        - (weight, pauli_dict) where:
            * weight: float, coefficient of the term
            * pauli_dict: dict, {site: 'X'/'Y'/'Z'/'I'}, e.g., {0: 'X', 1: 'Z'}
    
    Returns:
    --------
    hamiltonian : jVMC.operator.BranchFreeOperator
        The constructed Hamiltonian
    
    Examples:
    ---------
    # Example 1: Transverse field Ising model on 3 sites
    terms = [
        (-1.0, "ZZ", (0, 1)),
        (-1.0, "ZZ", (1, 2)),
        (-0.5, "X", (0,)),
        (-0.5, "X", (1,)),
        (-0.5, "X", (2,)),
    ]
    H = build_hamiltonian_from_pauli_strings(terms)
    
    # Example 2: Heisenberg model
    terms = [
        (1.0, "XX", (0, 1)),
        (1.0, "YY", (0, 1)),
        (1.0, "ZZ", (0, 1)),
    ]
    H = build_hamiltonian_from_pauli_strings(terms)
    
    # Example 3: Using dictionary format
    terms = [
        (-1.0, {0: 'Z', 1: 'Z', 2: 'Z'}),  # Three-body ZZZ term
        (-0.7, {0: 'X'}),                   # Single-body X term
    ]
    H = build_hamiltonian_from_pauli_strings(terms)
    
    # Example 4: Complex multi-body terms
    terms = [
        (0.5, "XYZZ", (0, 1, 2, 3)),       # 4-body term
        (-0.3, "XY", (5, 7)),               # Non-adjacent sites
        (1.0, "I", (0,)),                   # Identity (constant energy shift)
    ]
    H = build_hamiltonian_from_pauli_strings(terms)
    """
    
    # Operator mapping
    operator_map = {
        'X': jVMC.operator.Sx,
        'Y': jVMC.operator.Sy,
        'Z': jVMC.operator.Sz,
        'I': jVMC.operator.Id,  # Identity operator
    }
    
    # Initialize empty Hamiltonian
    hamiltonian = jVMC.operator.BranchFreeOperator()
    
    for term in pauli_terms:
        if len(term) == 2:
            # Dictionary format: (weight, pauli_dict)
            weight, pauli_dict = term
            
            # Sort by site index for consistency
            sorted_sites = sorted(pauli_dict.keys())
            ops = []
            
            for site in sorted_sites:
                pauli_op = pauli_dict[site].upper()
                if pauli_op not in operator_map:
                    raise ValueError(f"Invalid Pauli operator '{pauli_op}'. Must be X, Y, Z, or I.")
                ops.append(operator_map[pauli_op](site))
            
        elif len(term) == 3:
            # String format: (weight, pauli_string, sites)
            weight, pauli_string, sites = term
            
            # Validate input
            if len(pauli_string) != len(sites):
                raise ValueError(f"Length mismatch: pauli_string '{pauli_string}' has {len(pauli_string)} operators "
                               f"but sites {sites} has {len(sites)} elements.")
            
            ops = []
            for pauli_op, site in zip(pauli_string.upper(), sites):
                if pauli_op not in operator_map:
                    raise ValueError(f"Invalid Pauli operator '{pauli_op}'. Must be X, Y, Z, or I.")
                ops.append(operator_map[pauli_op](site))
        else:
            raise ValueError(f"Invalid term format. Expected (weight, pauli_string, sites) or (weight, pauli_dict), "
                           f"got tuple of length {len(term)}.")
        
        # Add term to Hamiltonian
        if len(ops) > 0:
            hamiltonian.add(jVMC.operator.scal_opstr(weight, tuple(ops)))
        else:
            # Handle constant terms (empty operator list)
            hamiltonian.add(jVMC.operator.scal_opstr(weight, ()))
    
    return hamiltonian


def print_hamiltonian_terms(pauli_terms):
    """
    Pretty print the Hamiltonian terms for verification.
    
    Parameters:
    -----------
    pauli_terms : list of tuples
        Same format as build_hamiltonian_from_pauli_strings
    """
    print("Hamiltonian Terms:")
    print("=" * 50)
    
    for i, term in enumerate(pauli_terms, 1):
        if len(term) == 2:
            weight, pauli_dict = term
            sorted_items = sorted(pauli_dict.items())
            term_str = " ⊗ ".join([f"{op}_{site}" for site, op in sorted_items])
        elif len(term) == 3:
            weight, pauli_string, sites = term
            term_str = " ⊗ ".join([f"{op}_{site}" for op, site in zip(pauli_string, sites)])
        
        sign = "+" if weight >= 0 else ""
        print(f"Term {i}: {sign}{weight:.4f} * {term_str}")
    
    print("=" * 50)
