#!/usr/bin/env python3
"""
Generate Hybrid MPI+OpenMP Architecture Diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_aspect('equal')

# Colors
COLOR_MPI = '#3498db'       # Blue for MPI ranks
COLOR_OMP = '#2ecc71'       # Green for OpenMP threads
COLOR_HALO = '#e74c3c'      # Red for Halo Exchange
COLOR_DOMAIN = '#ecf0f1'    # Light gray for global domain

# Title
ax.text(6, 6.5, 'Hybrid MPI + OpenMP Parallelism Architecture', 
        fontsize=16, fontweight='bold', ha='center', va='center')

# Global Domain Box
global_box = FancyBboxPatch((0.5, 1.5), 11, 4.5, 
                             boxstyle="round,pad=0.05", 
                             facecolor=COLOR_DOMAIN, edgecolor='black', linewidth=2)
ax.add_patch(global_box)
ax.text(6, 5.7, 'Global Problem Domain', fontsize=12, ha='center', style='italic')

# MPI Ranks (3 ranks)
rank_width = 3.2
rank_height = 3.5
rank_y = 1.8
rank_positions = [1, 4.4, 7.8]

for i, x in enumerate(rank_positions):
    # MPI Rank Box
    rank_box = FancyBboxPatch((x, rank_y), rank_width, rank_height,
                               boxstyle="round,pad=0.03",
                               facecolor=COLOR_MPI, edgecolor='black', 
                               linewidth=2, alpha=0.8)
    ax.add_patch(rank_box)
    ax.text(x + rank_width/2, rank_y + rank_height - 0.3, 
            f'MPI Rank {i}', fontsize=11, fontweight='bold', 
            ha='center', va='top', color='white')
    
    # OpenMP Threads inside each rank (2 threads)
    thread_width = 1.3
    thread_height = 1.8
    thread_y = rank_y + 0.4
    for j, tx in enumerate([x + 0.3, x + 1.6]):
        thread_box = FancyBboxPatch((tx, thread_y), thread_width, thread_height,
                                     boxstyle="round,pad=0.02",
                                     facecolor=COLOR_OMP, edgecolor='black',
                                     linewidth=1.5, alpha=0.9)
        ax.add_patch(thread_box)
        ax.text(tx + thread_width/2, thread_y + thread_height/2,
                f'Thread\n{j}', fontsize=9, ha='center', va='center', 
                color='white', fontweight='bold')

# Halo Exchange Arrows (between ranks)
arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=6"
for i in range(len(rank_positions) - 1):
    x1 = rank_positions[i] + rank_width
    x2 = rank_positions[i + 1]
    y = rank_y + rank_height / 2
    
    # Right arrow
    ax.annotate('', xy=(x2 - 0.1, y + 0.3), xytext=(x1 + 0.1, y + 0.3),
                arrowprops=dict(arrowstyle='->', color=COLOR_HALO, lw=2))
    # Left arrow
    ax.annotate('', xy=(x1 + 0.1, y - 0.3), xytext=(x2 - 0.1, y - 0.3),
                arrowprops=dict(arrowstyle='->', color=COLOR_HALO, lw=2))
    
    # Label
    ax.text((x1 + x2) / 2, y, 'Halo\nExchange', fontsize=8, ha='center', va='center',
            color=COLOR_HALO, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLOR_MPI, edgecolor='black', label='MPI Process (Inter-Node Communication)'),
    mpatches.Patch(facecolor=COLOR_OMP, edgecolor='black', label='OpenMP Thread (Shared Memory Parallelism)'),
    mpatches.Patch(facecolor=COLOR_HALO, edgecolor='black', label='Halo Exchange (MPI_Isend/Irecv)')
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9, 
          framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

# Annotations
ax.text(0.5, 0.8, '• MPI handles domain decomposition across nodes\n'
                  '• OpenMP parallelizes compute loops within each process\n'
                  '• Threads share L2/L3 cache → reduces memory bandwidth pressure',
        fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('hybrid_architecture.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: hybrid_architecture.png")
