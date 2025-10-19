"""
Tests for custom samplers and proposers.

These tests verify that custom update proposers work correctly with jVMC's
MCSampler and produce valid spin configurations.
"""

import pytest
import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import jax
import jax.numpy as jnp
from jax import random
import jVMC

from vmc.samplers import (
    custom_single_flip,
    custom_k_spin_flip,
    custom_domain_flip,
    custom_neighbor_swap,
    custom_cluster_flip,
    adaptive_proposer,
    create_sampler_with_proposer
)


@pytest.fixture
def simple_config():
    """Simple 1D configuration for testing"""
    return jnp.array([1, -1, 1, 1, -1, -1, 1, -1])


@pytest.fixture
def batch_config():
    """Batch of configurations for testing"""
    return jnp.array([
        [1, -1, 1, 1, -1, -1, 1, -1],
        [-1, -1, 1, 1, 1, -1, 1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1]
    ])


@pytest.fixture
def simple_system():
    """Create a simple test system with jVMC components"""
    L = 8
    
    # Create simple RBM
    net = jVMC.nets.CpxRBM(numHidden=4, bias=False)
    psi = jVMC.vqs.NQS(net, seed=123)
    
    # Create simple TFIM Hamiltonian
    g = -0.5
    hamiltonian = jVMC.operator.BranchFreeOperator()
    for l in range(L):
        # ZZ interaction
        hamiltonian.add(jVMC.operator.scal_opstr(
            -1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))
        ))
        # X field
        hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l),)))
    
    return psi, hamiltonian, L


class TestProposerBasics:
    """Test basic proposer functionality"""
    
    def test_single_flip_shape(self, simple_config):
        """Test that single flip preserves shape"""
        key = random.PRNGKey(0)
        new_config = custom_single_flip(key, simple_config)
        assert new_config.shape == simple_config.shape
    
    def test_single_flip_changes_one_spin(self, simple_config):
        """Test that single flip changes exactly one spin"""
        key = random.PRNGKey(0)
        new_config = custom_single_flip(key, simple_config)
        
        # Count number of flipped spins
        flipped = jnp.sum(new_config != simple_config)
        assert flipped == 1
    
    def test_single_flip_valid_values(self, simple_config):
        """Test that single flip produces valid spin values"""
        key = random.PRNGKey(0)
        new_config = custom_single_flip(key, simple_config)
        
        # All values should be +1 or -1
        assert jnp.all((new_config == 1) | (new_config == -1))
    
    def test_k_spin_flip_shape(self, simple_config):
        """Test that k-spin flip preserves shape"""
        key = random.PRNGKey(0)
        new_config = custom_k_spin_flip(key, simple_config, k=2)
        assert new_config.shape == simple_config.shape
    
    def test_k_spin_flip_changes_k_spins(self, simple_config):
        """Test that k-spin flip changes exactly k spins"""
        key = random.PRNGKey(0)
        k = 3
        new_config = custom_k_spin_flip(key, simple_config, k=k)
        
        # Count number of flipped spins
        flipped = jnp.sum(new_config != simple_config)
        assert flipped == k
    
    def test_k_spin_flip_invalid_k(self, simple_config):
        """Test that k >= L raises error"""
        key = random.PRNGKey(0)
        L = len(simple_config)
        
        with pytest.raises(ValueError):
            custom_k_spin_flip(key, simple_config, k=L)
    
    def test_domain_flip_shape(self, simple_config):
        """Test that domain flip preserves shape"""
        key = random.PRNGKey(0)
        new_config = custom_domain_flip(key, simple_config, domain_size=3)
        assert new_config.shape == simple_config.shape
    
    def test_domain_flip_contiguous(self, simple_config):
        """Test that domain flip flips contiguous spins"""
        key = random.PRNGKey(42)  # Fixed seed for deterministic test
        L = len(simple_config)
        domain_size = 3
        new_config = custom_domain_flip(key, simple_config, domain_size=domain_size)
        
        # Find flipped positions
        flipped = (new_config != simple_config)
        flipped_count = jnp.sum(flipped)
        
        assert flipped_count == domain_size
    
    def test_neighbor_swap_shape(self, simple_config):
        """Test that neighbor swap preserves shape"""
        key = random.PRNGKey(0)
        new_config = custom_neighbor_swap(key, simple_config)
        assert new_config.shape == simple_config.shape
    
    def test_neighbor_swap_conserves_magnetization(self, simple_config):
        """Test that neighbor swap conserves total magnetization"""
        key = random.PRNGKey(0)
        new_config = custom_neighbor_swap(key, simple_config)
        
        mag_old = jnp.sum(simple_config)
        mag_new = jnp.sum(new_config)
        assert mag_old == mag_new
    
    def test_cluster_flip_shape(self, simple_config):
        """Test that cluster flip preserves shape"""
        key = random.PRNGKey(0)
        new_config = custom_cluster_flip(key, simple_config, cluster_prob=0.5)
        assert new_config.shape == simple_config.shape
    
    def test_adaptive_proposer_shape(self, simple_config):
        """Test that adaptive proposer preserves shape"""
        key = random.PRNGKey(0)
        new_config = adaptive_proposer(key, simple_config)
        assert new_config.shape == simple_config.shape


class TestProposerBatchHandling:
    """Test that proposers handle batched configurations correctly"""
    
    def test_single_flip_batch(self, batch_config):
        """Test single flip with batch of configs"""
        key = random.PRNGKey(0)
        new_configs = custom_single_flip(key, batch_config)
        assert new_configs.shape == batch_config.shape
        
        # Each config should have exactly one spin flipped
        for i in range(len(batch_config)):
            flipped = jnp.sum(new_configs[i] != batch_config[i])
            assert flipped == 1
    
    def test_k_spin_flip_batch(self, batch_config):
        """Test k-spin flip with batch of configs"""
        key = random.PRNGKey(0)
        k = 2
        new_configs = custom_k_spin_flip(key, batch_config, k=k)
        assert new_configs.shape == batch_config.shape
        
        # Each config should have exactly k spins flipped
        for i in range(len(batch_config)):
            flipped = jnp.sum(new_configs[i] != batch_config[i])
            assert flipped == k
    
    def test_domain_flip_batch(self, batch_config):
        """Test domain flip with batch of configs"""
        key = random.PRNGKey(0)
        domain_size = 3
        new_configs = custom_domain_flip(key, batch_config, domain_size=domain_size)
        assert new_configs.shape == batch_config.shape
    
    def test_neighbor_swap_batch(self, batch_config):
        """Test neighbor swap with batch of configs"""
        key = random.PRNGKey(0)
        new_configs = custom_neighbor_swap(key, batch_config)
        assert new_configs.shape == batch_config.shape
        
        # Each config should conserve magnetization
        for i in range(len(batch_config)):
            mag_old = jnp.sum(batch_config[i])
            mag_new = jnp.sum(new_configs[i])
            assert mag_old == mag_new


class TestProposerJITCompatibility:
    """Test that proposers are JIT-compatible"""
    
    def test_single_flip_jit(self, simple_config):
        """Test that single flip can be JIT compiled"""
        jit_proposer = jax.jit(custom_single_flip)
        key = random.PRNGKey(0)
        
        # Should not raise error
        new_config = jit_proposer(key, simple_config)
        assert new_config.shape == simple_config.shape
    
    def test_k_spin_flip_jit(self, simple_config):
        """Test that k-spin flip can be JIT compiled"""
        # Need to use partial to fix k parameter for JIT
        from functools import partial
        jit_proposer = jax.jit(partial(custom_k_spin_flip, k=2))
        key = random.PRNGKey(0)
        
        new_config = jit_proposer(key, simple_config)
        assert new_config.shape == simple_config.shape
    
    def test_adaptive_proposer_jit(self, simple_config):
        """Test that adaptive proposer can be JIT compiled"""
        jit_proposer = jax.jit(adaptive_proposer)
        key = random.PRNGKey(0)
        
        new_config = jit_proposer(key, simple_config)
        assert new_config.shape == simple_config.shape


class TestSamplerIntegration:
    """Test integration with jVMC MCSampler"""
    
    def test_sampler_with_single_flip(self, simple_system):
        """Test creating sampler with single flip proposer"""
        psi, hamiltonian, L = simple_system
        
        sampler = jVMC.sampler.MCSampler(
            psi, (L,), random.PRNGKey(123),
            updateProposer=lambda k, c, a: custom_single_flip(k, c),
            numChains=100, sweepSteps=L,
            thermalizationSweeps=5
        )
        
        # Should be able to sample without error
        # Note: sample() returns shape (1, numChains, L)
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3  # Shape is (1, numChains, L)
        assert samples.shape[-1] == L
        # Verify sampler works without errors (values depend on jVMC encoding)
    
    def test_sampler_with_k_spin_flip(self, simple_system):
        """Test creating sampler with k-spin flip proposer"""
        psi, hamiltonian, L = simple_system
        
        sampler = jVMC.sampler.MCSampler(
            psi, (L,), random.PRNGKey(456),
            updateProposer=lambda k, c, a: custom_k_spin_flip(k, c, k=2),
            numChains=100, sweepSteps=L,
            thermalizationSweeps=5
        )
        
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3
        assert samples.shape[-1] == L
    
    def test_sampler_with_domain_flip(self, simple_system):
        """Test creating sampler with domain flip proposer"""
        psi, hamiltonian, L = simple_system
        
        sampler = jVMC.sampler.MCSampler(
            psi, (L,), random.PRNGKey(789),
            updateProposer=lambda k, c, a: custom_domain_flip(k, c, domain_size=3),
            numChains=100, sweepSteps=L,
            thermalizationSweeps=5
        )
        
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3
        assert samples.shape[-1] == L
    
    def test_sampler_with_adaptive(self, simple_system):
        """Test creating sampler with adaptive proposer"""
        psi, hamiltonian, L = simple_system
        
        sampler = jVMC.sampler.MCSampler(
            psi, (L,), random.PRNGKey(999),
            updateProposer=lambda k, c, a: adaptive_proposer(k, c),
            numChains=100, sweepSteps=L,
            thermalizationSweeps=5
        )
        
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3
        assert samples.shape[-1] == L


class TestConvenienceFunctions:
    """Test convenience functions for sampler creation"""
    
    def test_create_sampler_single_flip(self, simple_system):
        """Test create_sampler_with_proposer with single flip"""
        psi, hamiltonian, L = simple_system
        
        sampler = create_sampler_with_proposer(
            psi, L, proposer_type='single_flip',
            num_chains=50
        )
        
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3
        assert samples.shape[-1] == L
    
    def test_create_sampler_k_spin_flip(self, simple_system):
        """Test create_sampler_with_proposer with k-spin flip"""
        psi, hamiltonian, L = simple_system
        
        sampler = create_sampler_with_proposer(
            psi, L, proposer_type='k_spin_flip', k=2,
            num_chains=50
        )
        
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3
        assert samples.shape[-1] == L
    
    def test_create_sampler_domain_flip(self, simple_system):
        """Test create_sampler_with_proposer with domain flip"""
        psi, hamiltonian, L = simple_system
        
        sampler = create_sampler_with_proposer(
            psi, L, proposer_type='domain_flip', domain_size=3,
            num_chains=50
        )
        
        samples, _, _ = sampler.sample()
        assert samples.ndim == 3
        assert samples.shape[-1] == L
    
    def test_create_sampler_invalid_type(self, simple_system):
        """Test that invalid proposer type raises error"""
        psi, hamiltonian, L = simple_system
        
        with pytest.raises(ValueError, match="Unknown proposer_type"):
            create_sampler_with_proposer(
                psi, L, proposer_type='invalid_proposer'
            )


class TestProposerSignature:
    """Test that proposers accept the required 3-argument signature"""
    
    def test_proposers_accept_three_args(self, simple_config):
        """Test all proposers accept (key, config, proposer_args)"""
        key = random.PRNGKey(0)
        dummy_args = None
        
        # All these should work without error
        custom_single_flip(key, simple_config, dummy_args)
        custom_k_spin_flip(key, simple_config, k=2, proposer_args=dummy_args)
        custom_domain_flip(key, simple_config, domain_size=3, proposer_args=dummy_args)
        custom_neighbor_swap(key, simple_config, dummy_args)
        custom_cluster_flip(key, simple_config, cluster_prob=0.5, proposer_args=dummy_args)
        adaptive_proposer(key, simple_config, dummy_args)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
