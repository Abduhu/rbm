"""
Custom RBM architectures for jVMC

This module provides utilities for creating Restricted Boltzmann Machines (RBMs)
with custom sparse connectivity patterns.
"""

import flax.linen as nn
import jax.numpy as jnp
from typing import List, Tuple

# Import connectivity patterns
from .connectivity import (
    fully_connected_pattern,
    local_connectivity_pattern,
    nearest_neighbor_pattern,
    stripe_pattern,
    random_sparse_pattern,
    ring_pattern,
    checkerboard_pattern,
    hierarchical_pattern
)


def create_sparse_rbm(num_visible: int, 
                      num_hidden: int,
                      connections: List[Tuple[int, int]],
                      bias: bool = False,
                      dtype=jnp.complex64):
    """
    Create an RBM with custom sparse connectivity.
    
    This function allows you to specify exactly which visible units connect to which
    hidden units, enabling architectures like locally-connected, nearest-neighbor,
    or any custom connectivity pattern.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units (system size L)
    num_hidden : int
        Number of hidden units
    connections : List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) tuples specifying which nodes connect.
        Examples:
            - [(0, 0), (1, 0), (2, 1), (3, 1)] - local connections
            - [(i, j) for i in range(L) for j in range(H)] - fully connected
            - [(i, i%H) for i in range(L)] - each visible connects to one hidden
    bias : bool, optional
        Whether to include bias terms (default: False)
    dtype : jax dtype, optional
        Data type for parameters (default: jnp.complex64)
    
    Returns
    -------
    SparseRBM
        A Flax module instance with the specified connectivity that can be wrapped
        in jVMC.vqs.NQS for use in variational quantum Monte Carlo simulations.
    
    Examples
    --------
    >>> # Nearest-neighbor connectivity
    >>> connections = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]
    >>> net = create_sparse_rbm(6, 3, connections)
    >>> psi = jVMC.vqs.NQS(net, seed=1234)
    
    >>> # Fully connected (equivalent to default CpxRBM)
    >>> connections = [(i, j) for i in range(L) for j in range(H)]
    >>> net = create_sparse_rbm(L, H, connections)
    
    >>> # Custom pattern
    >>> connections = [(0, 0), (0, 1), (1, 1), (2, 2), (3, 2)]
    >>> net = create_sparse_rbm(4, 3, connections)
    
    Notes
    -----
    - The connectivity pattern is fixed at creation time and enforced during forward passes
    - Weights corresponding to non-existent connections are masked to zero
    - When wrapping in jVMC.vqs.NQS, initialize with shape (batch, 1, L) not (batch, L)
    """
    
    # Create connectivity mask from connection list
    mask = jnp.zeros((num_visible, num_hidden))
    for v, h in connections:
        if v >= num_visible or h >= num_hidden:
            raise ValueError(f"Connection ({v}, {h}) out of bounds. "
                           f"Valid ranges: visible [0, {num_visible}), hidden [0, {num_hidden})")
        mask = mask.at[v, h].set(1.0)
    
    class SparseRBM(nn.Module):
        """RBM with fixed sparse connectivity pattern"""
        
        @nn.compact
        def __call__(self, x):
            """
            Forward pass with sparse connectivity.
            
            Parameters
            ----------
            x : jax.Array
                Input configuration, shape (batch, L) or (L,)
                
            Returns
            -------
            jax.Array
                Log-amplitude of the wave function
            """
            x = jnp.atleast_2d(x)
            batch_size, L = x.shape
            
            if L != num_visible:
                raise ValueError(f"Input size {L} doesn't match num_visible {num_visible}")
            
            # Initialize weight matrix
            w_full = self.param('kernel', 
                               nn.initializers.normal(stddev=0.1),
                               (num_visible, num_hidden),
                               dtype)
            
            # Apply fixed connectivity mask
            w = w_full * mask
            
            # Compute hidden activations: h = x @ w
            hidden = jnp.dot(x, w)
            
            # Add biases if enabled
            if bias:
                hidden_bias = self.param('hidden_bias', 
                                        nn.initializers.zeros,
                                        (num_hidden,),
                                        dtype)
                hidden = hidden + hidden_bias
            
            # Sum over hidden units: log(1 + exp(h))
            # Use logaddexp for numerical stability
            hidden_contrib = jnp.sum(jnp.logaddexp(0, hidden), axis=-1)
            
            # Visible bias contribution
            if bias:
                visible_bias = self.param('visible_bias',
                                         nn.initializers.zeros,
                                         (num_visible,),
                                         dtype)
                visible_contrib = jnp.dot(x, visible_bias)
            else:
                visible_contrib = 0.0
            
            # Total log-amplitude
            log_psi = visible_contrib + hidden_contrib
            
            return jnp.squeeze(log_psi)
    
    return SparseRBM()

