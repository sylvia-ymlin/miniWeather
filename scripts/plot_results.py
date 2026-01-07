import matplotlib.pyplot as plt
import numpy as np

# Data from PROJECT_ANALYSIS.md
ranks = np.array([1, 2, 3, 4])

# Strong Scaling Data (100x50 fixed problem size)
strong_times = np.array([4.95, 2.52, 1.92, 1.76])
strong_speedup = strong_times[0] / strong_times
strong_ideal_speedup = ranks
strong_efficiency = (strong_speedup / ranks) * 100.0

# Weak Scaling Data (100x50 per rank)
weak_times = np.array([12.70, 25.03, 23.56, 38.31])
weak_efficiency = (weak_times[0] / weak_times) * 100.0

# Setup 1x2 Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Strong Scaling Efficiency
ax1.plot(ranks, strong_efficiency, 'o-', linewidth=2, color='#1f77b4', label='Measured')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal')
ax1.set_title('Strong Scaling Efficiency\n(Fixed Problem Size 100x50)', fontsize=12, fontweight='bold')
ax1.set_xlabel('MPI Ranks', fontsize=10)
ax1.set_ylabel('Efficiency (%)', fontsize=10)
ax1.set_ylim(0, 110)
ax1.set_xticks(ranks)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# Add annotations
for i, txt in enumerate(strong_efficiency):
    ax1.annotate(f"{txt:.1f}%", (ranks[i], strong_efficiency[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

# Plot 2: Weak Scaling Efficiency
ax2.plot(ranks, weak_efficiency, 's-', linewidth=2, color='#ff7f0e', label='Measured')
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal')
ax2.set_title('Weak Scaling Efficiency\n(Fixed Work per Rank 100x50)', fontsize=12, fontweight='bold')
ax2.set_xlabel('MPI Ranks', fontsize=10)
ax2.set_ylabel('Efficiency (%)', fontsize=10)
ax2.set_ylim(0, 110)
ax2.set_xticks(ranks)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

# Add annotations
for i, txt in enumerate(weak_efficiency):
    ax2.annotate(f"{txt:.1f}%", (ranks[i], weak_efficiency[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('scaling_results.png', dpi=300)
print("Plot saved to scaling_results.png")
