"""
Custom samplers and update proposers for jVMC Monte Carlo sampling.

This module provides various custom update proposers that can be used with
jVMC's MCSampler to implement different sampling strategies beyond the built-in
single spin flip.

All proposers follow the jVMC convention:
    proposer(key, config, proposer_args) -> new_config

where:
    - key: JAX random key for reproducibility
    - config: Current spin configurations (can be 1D or 2D array)
    - proposer_args: Optional additional arguments (often unused but required by jVMC)

Example Usage:
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> import jVMC
    >>> from vmc.samplers import custom_k_spin_flip, create_sampler_with_proposer
    >>> 
    >>> # Create a sampler with k-spin flip proposer
    >>> sampler = jVMC.sampler.MCSampler(
    ...     psi, (L,), random.PRNGKey(123),
    ...     updateProposer=lambda key, cfg, args: custom_k_spin_flip(key, cfg, k=2),
    ...     numChains=50, sweepSteps=L
    ... )
    >>> 
    >>> # Or use the convenience function
    >>> sampler = create_sampler_with_proposer(
    ...     psi, L, proposer_type='k_spin_flip', k=2
    ... )
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional, Callable


def custom_single_flip(key: jax.Array, config: jnp.ndarray, proposer_args=None) -> jnp.ndarray:
    """
    Single spin flip proposer (alternative implementation to jVMC's built-in).
    
    Randomly selects one spin and flips it: σᵢ → -σᵢ
    This is the most basic Metropolis update and works well for most systems.
    
    Args:
        key: JAX random key
        config: Current configuration(s), shape (L,) or (num_chains, L)
        proposer_args: Unused, but required by jVMC interface
        
    Returns:
        New configuration(s) with one randomly flipped spin
        
    Example:
        >>> config = jnp.array([1, -1, 1, 1])
        >>> new_config = custom_single_flip(random.PRNGKey(0), config)
        >>> # One spin will be flipped
    """
    config = jnp.atleast_2d(config)
    num_chains, L = config.shape
    
    keys = random.split(key, num_chains)
    
    def flip_single(cfg, key_i):
        idx = random.randint(key_i, (), 0, L)
        return cfg.at[idx].set(-cfg[idx])
    
    new_configs = jax.vmap(flip_single)(config, keys)
    return jnp.squeeze(new_configs)


def custom_k_spin_flip(key: jax.Array, config: jnp.ndarray, k: int = 2, 
                       proposer_args=None) -> jnp.ndarray:
    """
    Multi-spin flip proposer that flips k randomly selected spins.
    
    This proposer flips k distinct spins simultaneously, which can help the
    sampler escape local minima faster than single-spin flips. Useful for
    systems with strong correlations or large energy barriers.
    
    Args:
        key: JAX random key
        config: Current configuration(s), shape (L,) or (num_chains, L)
        k: Number of spins to flip simultaneously (default: 2)
        proposer_args: Unused, but required by jVMC interface
        
    Returns:
        New configuration(s) with k randomly flipped spins
        
    Example:
        >>> config = jnp.array([1, -1, 1, 1, -1, -1])
        >>> new_config = custom_k_spin_flip(random.PRNGKey(0), config, k=2)
        >>> # Two random spins will be flipped
        
    Note:
        - k should be less than L (system size)
        - Larger k means more aggressive exploration but lower acceptance
        - k=1 is equivalent to single spin flip
    """
    config = jnp.atleast_2d(config)
    num_chains, L = config.shape
    
    if k >= L:
        raise ValueError(f"k ({k}) must be less than system size L ({L})")
    
    keys = random.split(key, num_chains)
    
    def flip_k_spins(cfg, key_i):
        # Choose k distinct random positions
        indices = random.choice(key_i, L, shape=(k,), replace=False)
        # Create mask: 1 at selected positions, 0 elsewhere
        mask = jnp.zeros(L, dtype=cfg.dtype).at[indices].set(1)
        # Flip spins at masked positions: σᵢ → -σᵢ where mask=1
        return cfg * (1 - 2 * mask)
    
    new_configs = jax.vmap(flip_k_spins)(config, keys)
    return jnp.squeeze(new_configs)


def custom_domain_flip(key: jax.Array, config: jnp.ndarray, domain_size: int = 4,
                       proposer_args=None) -> jnp.ndarray:
    """
    Domain flip proposer that flips a contiguous region of spins.
    
    Flips a contiguous "domain" of spins, useful for systems with domain wall
    excitations (e.g., Ising model). This mimics physical domain wall dynamics
    and can be more efficient than random k-spin flips.
    
    Args:
        key: JAX random key
        config: Current configuration(s), shape (L,) or (num_chains, L)
        domain_size: Size of the contiguous domain to flip (default: 4)
        proposer_args: Unused, but required by jVMC interface
        
    Returns:
        New configuration(s) with a contiguous domain flipped
        
    Example:
        >>> config = jnp.array([1, 1, 1, -1, -1, -1, 1, 1])
        >>> new_config = custom_domain_flip(random.PRNGKey(0), config, domain_size=3)
        >>> # Three consecutive spins will be flipped (wraps around boundaries)
        
    Note:
        - Uses periodic boundary conditions
        - domain_size should be less than L
        - Good for systems with domain wall physics
    """
    config = jnp.atleast_2d(config)
    num_chains, L = config.shape
    
    if domain_size >= L:
        raise ValueError(f"domain_size ({domain_size}) must be less than system size L ({L})")
    
    keys = random.split(key, num_chains)
    
    def flip_domain(cfg, key_i):
        # Choose random starting position
        start_idx = random.randint(key_i, (), 0, L)
        # Create contiguous indices (with periodic boundary)
        indices = (start_idx + jnp.arange(domain_size)) % L
        # Create mask and flip
        mask = jnp.zeros(L, dtype=cfg.dtype).at[indices].set(1)
        return cfg * (1 - 2 * mask)
    
    new_configs = jax.vmap(flip_domain)(config, keys)
    return jnp.squeeze(new_configs)


def custom_neighbor_swap(key: jax.Array, config: jnp.ndarray, 
                         proposer_args=None) -> jnp.ndarray:
    """
    Neighbor swap (exchange) proposer.
    
    Swaps the values of two neighboring sites. Particularly useful for:
    - Particle-conserving systems (maintains total magnetization)
    - Heisenberg models
    - Systems where conservation laws are important
    
    Args:
        key: JAX random key
        config: Current configuration(s), shape (L,) or (num_chains, L)
        proposer_args: Unused, but required by jVMC interface
        
    Returns:
        New configuration(s) with two neighboring spins swapped
        
    Example:
        >>> config = jnp.array([1, -1, 1, 1, -1, -1])
        >>> new_config = custom_neighbor_swap(random.PRNGKey(0), config)
        >>> # Two neighboring spins will be swapped
        
    Note:
        - Uses periodic boundary conditions
        - Conserves total magnetization
        - May have lower acceptance rate for uniformly magnetized states
    """
    config = jnp.atleast_2d(config)
    num_chains, L = config.shape
    
    keys = random.split(key, num_chains)
    
    def swap_neighbors(cfg, key_i):
        # Choose random site
        i = random.randint(key_i, (), 0, L)
        j = (i + 1) % L  # Next neighbor with periodic boundary
        
        # Swap values
        new_cfg = cfg.at[i].set(cfg[j])
        new_cfg = new_cfg.at[j].set(cfg[i])
        return new_cfg
    
    new_configs = jax.vmap(swap_neighbors)(config, keys)
    return jnp.squeeze(new_configs)


def custom_cluster_flip(key: jax.Array, config: jnp.ndarray, cluster_prob: float = 0.5,
                        proposer_args=None) -> jnp.ndarray:
    """
    Simple cluster flip proposer (JAX-compatible simplified version).
    
    Flips a contiguous cluster of spins of random size. The cluster is flipped
    with probability cluster_prob. This is a simplified, JAX-compatible version
    that avoids control flow issues.
    
    Args:
        key: JAX random key
        config: Current configuration(s), shape (L,) or (num_chains, L)
        cluster_prob: Probability of actually flipping the cluster (default: 0.5)
        proposer_args: Unused, but required by jVMC interface
        
    Returns:
        New configuration(s) with cluster flipped
        
    Example:
        >>> config = jnp.array([1, 1, 1, -1, -1, 1])
        >>> new_config = custom_cluster_flip(random.PRNGKey(0), config, cluster_prob=0.3)
        >>> # A contiguous cluster of spins may be flipped
        
    Note:
        - cluster_prob controls flip probability, not cluster growth
        - Cluster size is random between 1 and L//2
        - Simplified for JAX compatibility (no conditional control flow)
    """
    config = jnp.atleast_2d(config)
    num_chains, L = config.shape
    
    keys = random.split(key, num_chains)
    
    def flip_cluster(cfg, key_i):
        # Split keys for reproducibility
        key_start, key_size, key_flip = random.split(key_i, 3)
        
        # Choose random start site
        start_site = random.randint(key_start, (), 0, L)
        
        # Generate cluster size as a probability mask (JAX-traceable)
        # For each position, decide if it's in the cluster with decreasing probability
        max_cluster_size = L // 2
        
        # Create all possible indices from start site
        all_indices = (start_site + jnp.arange(L)) % L
        
        # Generate random values for each position
        rand_vals = random.uniform(key_size, (L,))
        
        # Positions are in cluster if their random value < cluster_prob
        # and they're within max_cluster_size from start
        position_offset = jnp.arange(L)
        in_range = position_offset < max_cluster_size
        in_cluster = rand_vals < cluster_prob
        cluster = in_range & in_cluster
        
        # Apply cluster mask to configuration
        flipped = jnp.where(cluster, -cfg[all_indices], cfg[all_indices])
        
        # Reconstruct configuration with flipped cluster
        result = cfg.at[all_indices].set(jnp.where(cluster, flipped, cfg[all_indices]))
        
        # Decide whether to actually apply the flip
        apply_flip = random.uniform(key_flip, ()) < cluster_prob
        return jnp.where(apply_flip, result, cfg)
    
    new_configs = jax.vmap(flip_cluster)(config, keys)
    return jnp.squeeze(new_configs)


def adaptive_proposer(key: jax.Array, config: jnp.ndarray, 
                      proposer_args=None) -> jnp.ndarray:
    """
    Adaptive proposer that randomly selects between different move types.
    
    Combines multiple proposer strategies:
    - 70% probability: single spin flip (local exploration)
    - 20% probability: 2-spin flip (medium exploration)
    - 10% probability: domain flip (size=3, larger exploration)
    
    This adaptive strategy balances local refinement with global exploration,
    often improving mixing and reducing autocorrelation times.
    
    Args:
        key: JAX random key
        config: Current configuration(s), shape (L,) or (num_chains, L)
        proposer_args: Unused, but required by jVMC interface
        
    Returns:
        New configuration(s) using a randomly selected move type
        
    Example:
        >>> config = jnp.array([1, -1, 1, 1, -1, -1])
        >>> new_config = adaptive_proposer(random.PRNGKey(0), config)
        >>> # One of three move types will be applied
        
    Note:
        - Move probabilities are inlined for JIT compatibility
        - All logic is differentiable and JIT-safe
        - Can be customized by modifying probability thresholds
    """
    config = jnp.atleast_2d(config)
    num_chains, L = config.shape
    
    # Split keys: one for move selection, rest for execution
    key_select, key_exec = random.split(key, 2)
    keys_chains = random.split(key_exec, num_chains)
    
    # Generate random move type for each chain
    move_rands = random.uniform(key_select, (num_chains,))
    
    def apply_move(cfg, key_i, rand):
        """Apply move based on random value"""
        # Split key for different operations
        key1, key2, key3 = random.split(key_i, 3)
        
        # Move 1: Single flip
        idx1 = random.randint(key1, (), 0, L)
        move1 = cfg.at[idx1].set(-cfg[idx1])
        
        # Move 2: 2-spin flip
        indices2 = random.choice(key2, L, shape=(2,), replace=False)
        mask2 = jnp.zeros(L, dtype=cfg.dtype).at[indices2].set(1)
        move2 = cfg * (1 - 2 * mask2)
        
        # Move 3: Domain flip (size=3)
        start_idx = random.randint(key3, (), 0, L)
        indices3 = (start_idx + jnp.arange(3)) % L
        mask3 = jnp.zeros(L, dtype=cfg.dtype).at[indices3].set(1)
        move3 = cfg * (1 - 2 * mask3)
        
        # Select based on random value (70% move1, 20% move2, 10% move3)
        result = jax.lax.cond(
            rand < 0.7,
            lambda _: move1,
            lambda _: jax.lax.cond(
                rand < 0.9,
                lambda _: move2,
                lambda _: move3,
                operand=None
            ),
            operand=None
        )
        return result
    
    new_configs = jax.vmap(apply_move)(config, keys_chains, move_rands)
    return jnp.squeeze(new_configs)


def create_sampler_with_proposer(
    psi,
    system_size: int,
    proposer_type: str = 'single_flip',
    seed: int = 123,
    num_chains: int = 50,
    num_samples: int = 1000,
    thermalization_sweeps: int = 10,
    **proposer_kwargs
):
    """
    Convenience function to create a jVMC sampler with a custom proposer.
    
    Args:
        psi: jVMC NQS wavefunction
        system_size: System size L
        proposer_type: Type of proposer to use. Options:
            - 'single_flip': Single spin flip
            - 'k_spin_flip': k-spin flip (requires k parameter)
            - 'domain_flip': Contiguous domain flip (requires domain_size parameter)
            - 'neighbor_swap': Neighbor swap
            - 'cluster_flip': Cluster flip (requires cluster_prob parameter)
            - 'adaptive': Adaptive multi-strategy proposer
        seed: Random seed for reproducibility
        num_chains: Number of parallel Markov chains
        num_samples: Number of samples to generate
        thermalization_sweeps: Number of thermalization sweeps
        **proposer_kwargs: Additional parameters for the proposer
        
    Returns:
        Configured jVMC MCSampler instance
        
    Example:
        >>> import jVMC
        >>> from vmc.samplers import create_sampler_with_proposer
        >>> 
        >>> # Create sampler with k-spin flip
        >>> sampler = create_sampler_with_proposer(
        ...     psi, L=10, proposer_type='k_spin_flip', k=2,
        ...     num_chains=100, num_samples=2000
        ... )
        >>> samples, _, _ = sampler.sample()
    """
    import jVMC
    from jax import random as jax_random
    
    # Map proposer types to functions
    proposer_map = {
        'single_flip': lambda k, c, a: custom_single_flip(k, c),
        'k_spin_flip': lambda k, c, a: custom_k_spin_flip(k, c, **proposer_kwargs),
        'domain_flip': lambda k, c, a: custom_domain_flip(k, c, **proposer_kwargs),
        'neighbor_swap': lambda k, c, a: custom_neighbor_swap(k, c),
        'cluster_flip': lambda k, c, a: custom_cluster_flip(k, c, **proposer_kwargs),
        'adaptive': lambda k, c, a: adaptive_proposer(k, c),
    }
    
    if proposer_type not in proposer_map:
        raise ValueError(
            f"Unknown proposer_type: {proposer_type}. "
            f"Available options: {list(proposer_map.keys())}"
        )
    
    proposer_func = proposer_map[proposer_type]
    
    # Create sampler
    sampler = jVMC.sampler.MCSampler(
        psi,
        (system_size,),
        jax_random.PRNGKey(seed),
        updateProposer=proposer_func,
        numChains=num_chains,
        sweepSteps=system_size,
        numSamples=num_samples,
        thermalizationSweeps=thermalization_sweeps
    )
    
    return sampler


def test_proposer_efficiency(
    proposer: Callable,
    proposer_name: str,
    psi,
    hamiltonian,
    system_size: int,
    seed: int = 1234,
    num_chains: int = 50,
    num_samples: int = 2000
) -> dict:
    """
    Test a proposer and report sampling statistics.
    
    Useful for comparing different proposers and tuning parameters.
    
    Args:
        proposer: Proposer function with signature (key, config, args) -> new_config
        proposer_name: Name for display in results
        psi: jVMC NQS wavefunction
        hamiltonian: jVMC Hamiltonian operator
        system_size: System size L
        seed: Random seed
        num_chains: Number of parallel chains
        num_samples: Number of samples to generate
        
    Returns:
        Dictionary with statistics:
            - 'diversity': Fraction of unique samples
            - 'energy': Mean energy per site
            - 'variance': Energy variance per site
            - 'std': Energy standard deviation per site
            - 'num_unique': Number of unique samples
            - 'total_samples': Total number of samples
            
    Example:
        >>> from vmc.samplers import test_proposer_efficiency, custom_k_spin_flip
        >>> 
        >>> stats = test_proposer_efficiency(
        ...     lambda k, c, a: custom_k_spin_flip(k, c, k=2),
        ...     "2-Spin Flip",
        ...     psi, hamiltonian, L=10
        ... )
        >>> print(f"Energy: {stats['energy']:.6f} ± {stats['std']:.6f}")
    """
    import jVMC
    from jax import random as jax_random
    
    # Create sampler
    sampler = jVMC.sampler.MCSampler(
        psi, (system_size,), jax_random.PRNGKey(seed),
        updateProposer=proposer,
        numChains=num_chains,
        sweepSteps=system_size,
        thermalizationSweeps=20
    )
    
    # Sample (pass numSamples to sample() method)
    samples, _, _ = sampler.sample(numSamples=num_samples)
    
    # Compute statistics
    unique_samples = jnp.unique(samples, axis=0).shape[0]
    total_samples = samples.shape[0]
    diversity = unique_samples / total_samples
    
    # Energy statistics
    Eloc = hamiltonian.get_O_loc(samples, psi)
    E_mean = jnp.mean(jnp.real(Eloc)) / system_size
    E_var = jnp.var(jnp.real(Eloc)) / system_size
    E_std = jnp.sqrt(E_var)
    
    print(f"\n{proposer_name}:")
    print(f"  Unique samples: {unique_samples}/{total_samples} ({diversity:.2%})")
    print(f"  Energy/site: {E_mean:.6f} ± {E_std:.6f}")
    print(f"  Variance/site: {E_var:.6f}")
    
    return {
        'diversity': float(diversity),
        'energy': float(E_mean),
        'variance': float(E_var),
        'std': float(E_std),
        'num_unique': int(unique_samples),
        'total_samples': int(total_samples)
    }
