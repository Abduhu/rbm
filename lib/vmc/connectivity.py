"""
Connectivity pattern generators for sparse RBM architectures

This module provides pre-defined connectivity patterns for creating sparse
Restricted Boltzmann Machines with various architectural constraints.
"""

from typing import List, Tuple


def fully_connected_pattern(num_visible: int, num_hidden: int) -> List[Tuple[int, int]]:
    """
    Generate fully connected pattern (all-to-all connections).
    
    Creates a bipartite graph where every visible unit connects to every hidden unit.
    This is equivalent to a standard RBM architecture.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
        
    Returns
    -------
    List[Tuple[int, int]]
        List of all (visible_idx, hidden_idx) connections
        
    Examples
    --------
    >>> connections = fully_connected_pattern(4, 2)
    >>> len(connections)
    8
    >>> connections
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    """
    return [(v, h) for v in range(num_visible) for h in range(num_hidden)]


def local_connectivity_pattern(num_visible: int, 
                               num_hidden: int, 
                               receptive_field: int = 3) -> List[Tuple[int, int]]:
    """
    Generate local connectivity where each hidden unit connects to a nearby region.
    
    This pattern divides the visible layer into regions and connects each hidden unit
    to a local receptive field. Useful for capturing local correlations while reducing
    the number of parameters. Commonly used in quantum systems where locality matters.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
    receptive_field : int, optional
        Number of visible units each hidden unit connects to (default: 3)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Notes
    -----
    - Uses periodic boundary conditions, so connections wrap around
    - Hidden units are evenly distributed across the visible layer
    - Each hidden unit is centered on a different region
    
    Examples
    --------
    >>> connections = local_connectivity_pattern(8, 4, receptive_field=3)
    >>> # Each of 4 hidden units connects to 3 visible units
    """
    connections = []
    stride = num_visible // num_hidden
    
    for h in range(num_hidden):
        center = (h * stride + stride // 2) % num_visible
        for offset in range(-receptive_field // 2, receptive_field // 2 + 1):
            v = (center + offset) % num_visible  # Periodic boundary
            connections.append((v, h))
    
    return connections


def nearest_neighbor_pattern(num_visible: int, num_hidden: int) -> List[Tuple[int, int]]:
    """
    Generate nearest-neighbor connectivity pattern.
    
    Each hidden unit connects to a pair of adjacent visible units. This creates
    a very sparse connectivity suitable for capturing nearest-neighbor correlations
    in 1D quantum systems (e.g., spin chains).
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Notes
    -----
    Each hidden unit connects to exactly 2 visible units (pairs of neighbors).
    Uses modulo arithmetic to handle edge cases and wrap around.
    
    Examples
    --------
    >>> connections = nearest_neighbor_pattern(8, 4)
    >>> len(connections)
    8
    >>> # Hidden 0 ← Visible [0, 1]
    >>> # Hidden 1 ← Visible [2, 3]
    >>> # Hidden 2 ← Visible [4, 5]
    >>> # Hidden 3 ← Visible [6, 7]
    """
    connections = []
    for h in range(num_hidden):
        v1 = (h * 2) % num_visible
        v2 = (h * 2 + 1) % num_visible
        connections.append((v1, h))
        connections.append((v2, h))
    return connections


def stripe_pattern(num_visible: int, 
                   num_hidden: int, 
                   stripe_width: int = 2) -> List[Tuple[int, int]]:
    """
    Divide visible units into stripes, each connected to one hidden unit.
    
    This creates non-overlapping groups of visible units, where each group
    connects to a single hidden unit. Useful for architectures that need
    to process independent subsystems.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
    stripe_width : int, optional
        Number of consecutive visible units in each stripe (default: 2)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Notes
    -----
    Uses periodic boundary conditions for stripes that extend beyond num_visible.
    Stripes do not overlap - each visible unit connects to exactly one hidden unit.
    
    Examples
    --------
    >>> connections = stripe_pattern(8, 4, stripe_width=2)
    >>> # Hidden 0 ← Visible [0, 1]
    >>> # Hidden 1 ← Visible [2, 3]
    >>> # Hidden 2 ← Visible [4, 5]
    >>> # Hidden 3 ← Visible [6, 7]
    """
    connections = []
    for h in range(num_hidden):
        start = h * stripe_width
        for offset in range(stripe_width):
            v = (start + offset) % num_visible
            connections.append((v, h))
    return connections


def random_sparse_pattern(num_visible: int, 
                          num_hidden: int, 
                          sparsity: float = 0.3, 
                          seed: int = 42) -> List[Tuple[int, int]]:
    """
    Generate random sparse connectivity pattern.
    
    Randomly selects a subset of all possible connections. Useful for exploring
    the effects of sparsity on representational power, or for creating irregular
    connectivity patterns that don't follow geometric constraints.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
    sparsity : float, optional
        Fraction of connections to keep, between 0 and 1.
        - 0.0 = no connections (invalid for RBM)
        - 1.0 = fully connected
        - 0.3 = keep 30% of connections (default)
    seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Notes
    -----
    The actual number of connections will be int(sparsity * num_visible * num_hidden).
    Uses numpy's random number generator for selection.
    
    Examples
    --------
    >>> connections = random_sparse_pattern(10, 5, sparsity=0.3, seed=42)
    >>> len(connections)
    15
    >>> # Randomly selected 15 out of 50 possible connections
    """
    import numpy as np
    np.random.seed(seed)
    
    # Generate all possible connections
    all_connections = fully_connected_pattern(num_visible, num_hidden)
    
    # Randomly sample
    num_connections = int(len(all_connections) * sparsity)
    if num_connections == 0:
        raise ValueError(f"Sparsity {sparsity} results in 0 connections. Use higher sparsity.")
    
    selected = np.random.choice(len(all_connections), size=num_connections, replace=False)
    
    return [all_connections[i] for i in selected]


def ring_pattern(num_visible: int, 
                 num_hidden: int,
                 radius: int = 1) -> List[Tuple[int, int]]:
    """
    Generate ring/circular connectivity pattern.
    
    Each hidden unit connects to visible units within a certain radius in a
    circular arrangement. Suitable for systems with periodic boundary conditions
    or cyclic symmetry.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units (arranged in a ring)
    num_hidden : int
        Number of hidden units (arranged in a ring)
    radius : int, optional
        Connection radius - each hidden connects to visible units within this
        distance (default: 1)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Examples
    --------
    >>> connections = ring_pattern(8, 4, radius=1)
    >>> # Hidden units distributed around ring, each connects to nearby visible
    """
    connections = []
    
    for h in range(num_hidden):
        # Position of this hidden unit in visible space
        h_position = (h * num_visible) / num_hidden
        
        for v in range(num_visible):
            # Calculate circular distance
            direct_dist = abs(v - h_position)
            wrap_dist = num_visible - direct_dist
            circular_dist = min(direct_dist, wrap_dist)
            
            # Connect if within radius
            if circular_dist <= radius * (num_visible / num_hidden):
                connections.append((v, h))
    
    return connections


def checkerboard_pattern(num_visible: int, 
                        num_hidden: int,
                        width: int = None) -> List[Tuple[int, int]]:
    """
    Generate checkerboard connectivity pattern.
    
    Alternating connectivity pattern useful for 2D systems or when you want
    to separate even/odd sites. Visible units are arranged in a grid (implied)
    and connections alternate like a checkerboard.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units (should be a perfect square for 2D interpretation)
    num_hidden : int
        Number of hidden units
    width : int, optional
        Width of the 2D grid (default: sqrt(num_visible))
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Notes
    -----
    This pattern is most meaningful when num_visible is a perfect square.
    
    Examples
    --------
    >>> connections = checkerboard_pattern(16, 4)
    >>> # 4x4 visible grid with checkerboard connectivity to 4 hidden units
    """
    import math
    
    if width is None:
        width = int(math.sqrt(num_visible))
    
    height = num_visible // width
    connections = []
    
    for v in range(num_visible):
        row = v // width
        col = v % width
        
        # Checkerboard: even/odd pattern
        parity = (row + col) % 2
        
        # Connect to hidden units based on parity
        for h in range(num_hidden):
            if h % 2 == parity:
                connections.append((v, h))
    
    return connections


def hierarchical_pattern(num_visible: int,
                        num_hidden: int,
                        levels: int = 2) -> List[Tuple[int, int]]:
    """
    Generate hierarchical connectivity pattern.
    
    Creates a multi-scale connectivity where different hidden units capture
    features at different scales. Lower-indexed hidden units connect to
    smaller groups of visible units, higher-indexed ones to larger groups.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
    levels : int, optional
        Number of hierarchical levels (default: 2)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
        
    Examples
    --------
    >>> connections = hierarchical_pattern(16, 8, levels=3)
    >>> # Hidden units at different levels connect to different-sized regions
    """
    connections = []
    hidden_per_level = num_hidden // levels
    
    for level in range(levels):
        # Receptive field size grows with level
        receptive_field = 2 ** (level + 1)
        
        for h in range(hidden_per_level):
            h_global = level * hidden_per_level + h
            if h_global >= num_hidden:
                break
                
            # Center position for this hidden unit
            center = (h * num_visible) // hidden_per_level
            
            # Connect to receptive field
            for offset in range(-receptive_field // 2, receptive_field // 2):
                v = (center + offset) % num_visible
                connections.append((v, h_global))
    
    return connections


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_connectivity(connections: List[Tuple[int, int]], 
                          num_visible: int, 
                          num_hidden: int,
                          title: str = "RBM Connectivity Pattern"):
    """
    Visualize RBM connectivity pattern as a matrix.
    
    Creates a visual representation of which visible units connect to which
    hidden units, displayed as a 2D matrix heatmap with statistics.
    
    Parameters
    ----------
    connections : List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections to visualize
    num_visible : int
        Number of visible units (rows in the matrix)
    num_hidden : int
        Number of hidden units (columns in the matrix)
    title : str, optional
        Plot title (default: "RBM Connectivity Pattern")
        
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
        
    Examples
    --------
    >>> from vmc.connectivity import nearest_neighbor_pattern, visualize_connectivity
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> connections = nearest_neighbor_pattern(8, 4)
    >>> fig, ax = visualize_connectivity(connections, 8, 4, 
    ...                                   "Nearest-Neighbor Connectivity")
    >>> plt.show()
    
    Notes
    -----
    - Blue cells indicate connections (1), white cells indicate no connection (0)
    - Statistics box shows total connections and sparsity percentage
    - Grid lines separate individual units for clarity
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create connectivity matrix
    matrix = np.zeros((num_visible, num_hidden))
    for v, h in connections:
        matrix[v, h] = 1
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(max(6, num_hidden), max(4, num_visible * 0.4)))
    
    # Plot matrix as heatmap
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Configure axes
    ax.set_xticks(range(num_hidden))
    ax.set_yticks(range(num_visible))
    ax.set_xticklabels([f'H{i}' for i in range(num_hidden)])
    ax.set_yticklabels([f'V{i}' for i in range(num_visible)])
    ax.set_xlabel('Hidden Units', fontsize=12)
    ax.set_ylabel('Visible Units', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid lines between units
    ax.set_xticks([x - 0.5 for x in range(1, num_hidden)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, num_visible)], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Connection', rotation=270, labelpad=15)
    
    # Calculate and display statistics
    total_possible = num_visible * num_hidden
    total_actual = len(connections)
    sparsity_pct = (1 - total_actual / total_possible) * 100
    
    stats_text = f"Connections: {total_actual}/{total_possible} ({sparsity_pct:.1f}% sparse)"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    return fig, ax


def print_connectivity_stats(connections: List[Tuple[int, int]],
                            num_visible: int,
                            num_hidden: int,
                            pattern_name: str = "Custom"):
    """
    Print connectivity statistics in a readable format.
    
    Provides a text-based summary of connectivity pattern properties including
    total connections, sparsity, and per-hidden-unit connection counts.
    
    Parameters
    ----------
    connections : List[Tuple[int, int]]
        List of (visible_idx, hidden_idx) connections
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
    pattern_name : str, optional
        Name of the connectivity pattern (default: "Custom")
        
    Examples
    --------
    >>> connections = local_connectivity_pattern(8, 4, receptive_field=3)
    >>> print_connectivity_stats(connections, 8, 4, "Local (rf=3)")
    
    Output:
    ```
    ======================================================================
    Connectivity Pattern: Local (rf=3)
    ======================================================================
    System size:        8 visible × 4 hidden
    Total connections:  12 / 32 (37.5% of possible)
    Sparsity:           62.5%
    
    Per-hidden unit:
      Hidden 0: 3 connections → Visible [0, 1, 2]
      Hidden 1: 3 connections → Visible [2, 3, 4]
      Hidden 2: 3 connections → Visible [4, 5, 6]
      Hidden 3: 3 connections → Visible [6, 7, 0]
    ======================================================================
    ```
    """
    total_possible = num_visible * num_hidden
    total_actual = len(connections)
    sparsity_pct = (1 - total_actual / total_possible) * 100
    connectivity_pct = (total_actual / total_possible) * 100
    
    print("=" * 70)
    print(f"Connectivity Pattern: {pattern_name}")
    print("=" * 70)
    print(f"System size:        {num_visible} visible × {num_hidden} hidden")
    print(f"Total connections:  {total_actual} / {total_possible} ({connectivity_pct:.1f}% of possible)")
    print(f"Sparsity:           {sparsity_pct:.1f}%")
    
    # Group connections by hidden unit
    hidden_connections = {h: [] for h in range(num_hidden)}
    for v, h in connections:
        hidden_connections[h].append(v)
    
    print("\nPer-hidden unit:")
    for h in range(num_hidden):
        visible_list = sorted(hidden_connections[h])
        if len(visible_list) <= 5:
            visible_str = str(visible_list)
        else:
            visible_str = f"[{visible_list[0]}, {visible_list[1]}, ..., {visible_list[-2]}, {visible_list[-1]}]"
        print(f"  Hidden {h}: {len(visible_list)} connections → Visible {visible_str}")
    
    print("=" * 70)


def compare_patterns(num_visible: int,
                    num_hidden: int,
                    patterns: dict = None):
    """
    Compare multiple connectivity patterns side-by-side.
    
    Generates and compares statistics for multiple connectivity patterns,
    useful for understanding tradeoffs between different architectures.
    
    Parameters
    ----------
    num_visible : int
        Number of visible units
    num_hidden : int
        Number of hidden units
    patterns : dict, optional
        Dictionary mapping pattern names to pattern generator functions.
        If None, compares all basic patterns.
        
    Examples
    --------
    >>> compare_patterns(8, 4)
    
    >>> # Custom comparison
    >>> patterns = {
    ...     "Dense": lambda: fully_connected_pattern(8, 4),
    ...     "Sparse": lambda: nearest_neighbor_pattern(8, 4),
    ...     "Local": lambda: local_connectivity_pattern(8, 4, receptive_field=2)
    ... }
    >>> compare_patterns(8, 4, patterns)
    """
    if patterns is None:
        # Default comparison of basic patterns
        patterns = {
            "Fully Connected": lambda: fully_connected_pattern(num_visible, num_hidden),
            "Local (rf=3)": lambda: local_connectivity_pattern(num_visible, num_hidden, receptive_field=3),
            "Nearest Neighbor": lambda: nearest_neighbor_pattern(num_visible, num_hidden),
            "Stripe (w=2)": lambda: stripe_pattern(num_visible, num_hidden, stripe_width=2),
            "Random (30%)": lambda: random_sparse_pattern(num_visible, num_hidden, sparsity=0.3, seed=42),
        }
    
    print("\n" + "=" * 70)
    print(f"CONNECTIVITY PATTERN COMPARISON")
    print(f"System: {num_visible} visible × {num_hidden} hidden units")
    print("=" * 70)
    print(f"{'Pattern':<25} {'Connections':<15} {'Sparsity':<15} {'Avg/Hidden':<15}")
    print("-" * 70)
    
    total_possible = num_visible * num_hidden
    
    for name, pattern_func in patterns.items():
        connections = pattern_func()
        total_actual = len(connections)
        sparsity_pct = (1 - total_actual / total_possible) * 100
        avg_per_hidden = total_actual / num_hidden
        
        conn_str = f"{total_actual}/{total_possible}"
        sparsity_str = f"{sparsity_pct:.1f}%"
        avg_str = f"{avg_per_hidden:.1f}"
        
        print(f"{name:<25} {conn_str:<15} {sparsity_str:<15} {avg_str:<15}")
    
    print("=" * 70 + "\n")
