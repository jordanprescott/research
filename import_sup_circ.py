import cotengra as ctg

# Use the ContractionTree optimizer
opt = ctg.HyperOptimizer(
    max_repeats=32,  # More repeats for better pathfinding
    parallel="ray",  # Use parallel computation if available
)

# Find the optimized contraction path
path, info = opt.optimize(circ.psi)

# Use the optimized path to perform the contraction
result = circ.psi.contract(optimal_path=path)
print(result)
