using Plots

# 1. Parameter Calibration (Rough estimates for an emerging market)
const β = 0.95      # Discount factor
const σ = 2.0       # Coefficient of relative risk aversion
const γ = 2.0       # Inverse Frisch elasticity (informal sector)
const ψ = 1.5       # Weight on informal/home production
const α = 0.3       # Output elasticity of public capital
const δ = 0.1       # Depreciation rate
const θ = 1.0       # Single productivity type for first-best benchmark

# 2. State and Choice Space Discretization
const N_K = 200     # Number of points on the Capital grid
const N_l = 100     # Number of points on the Informal Labor grid

# Create grids
const K_grid = range(0.1, stop=10.0, length=N_K)
const l_grid = range(0.01, stop=0.99, length=N_l)

# 3. Define Utility and Production Functions
function u(c)
    return c > 0 ? (c^(1 - σ)) / (1 - σ) : -1e10
end

function v(l)
    return ψ * (l^(1 - γ)) / (1 - γ)
end

function F(K, l)
    # Output: K^α * (θ * formal_labor)^(1-α)
    formal_labor = 1.0 - l
    return (K^α) * ((θ * formal_labor)^(1 - α))
end

# 4. Value Function Iteration (VFI) Setup
V_old = zeros(N_K)      # Initial guess for V(K)
V_new = zeros(N_K)
pol_K = zeros(Int, N_K) # To store indices of optimal K'
pol_l = zeros(Int, N_K) # To store indices of optimal l

const max_iter = 1000
const tol = 1e-6

println("Starting Value Function Iteration...")

# 5. The Main VFI Loop
for iter in 1:max_iter
    max_diff = 0.0
    
    # Loop over current capital states
    for (i, K) in enumerate(K_grid)
        best_val = -Inf
        best_K_idx = 1
        best_l_idx = 1
        
        # Grid search over next period capital (K')
        for (j, K_next) in enumerate(K_grid)
            # Grid search over informal labor (l)
            for (m, l) in enumerate(l_grid)
                
                # Resource Constraint: c = F(K, l) + (1-δ)K - K'
                c = F(K, l) + (1 - δ) * K - K_next
                
                if c > 0
                    # Current utility + discounted continuation value
                    val = u(c) + v(l) + β * V_old[j]
                    
                    if val > best_val
                        best_val = val
                        best_K_idx = j
                        best_l_idx = m
                    end
                end
            end
        end
        
        # Update the value and record the optimal policies
        V_new[i] = best_val
        pol_K[i] = best_K_idx
        pol_l[i] = best_l_idx
        
        # Track the maximum difference for convergence
        max_diff = max(max_diff, abs(V_new[i] - V_old[i]))
    end
    
    # Check convergence
    if max_diff < tol
        println("Converged in $iter iterations!")
        break
    end
    
    # Update V_old for the next iteration (the dot .= updates in-place, which is fast)
    V_old .= V_new
end

# 6. Extract the actual policy values from the indices
K_policy = [K_grid[pol_K[i]] for i in 1:N_K]
l_policy = [l_grid[pol_l[i]] for i in 1:N_K]
formal_labor_policy = 1.0 .- l_policy

# Calculate optimal consumption to plot
c_policy = zeros(N_K)
for i in 1:N_K
    c_policy[i] = F(K_grid[i], l_policy[i]) + (1 - δ) * K_grid[i] - K_policy[i]
end

# 7. Visualization
println("Generating plots...")

p1 = plot(K_grid, V_new, title="Value Function V(K)", xlabel="Public Capital (K)", ylabel="Utility", legend=false, lw=2)
p2 = plot(K_grid, K_policy, title="Capital Policy K'(K)", xlabel="Current K", ylabel="Next K'", legend=false, lw=2)
plot!(p2, K_grid, K_grid, linestyle=:dash, color=:gray) # Adds the 45-degree line to find steady state
p3 = plot(K_grid, c_policy, title="Consumption Policy", xlabel="Current K", ylabel="Formal Consumption", legend=false, lw=2, color=:green)
p4 = plot(K_grid, formal_labor_policy, title="Formal Labor (1-l)", xlabel="Current K", ylabel="Labor Effort", legend=false, lw=2, color=:orange)

# Combine all 4 plots into one window
display(plot(p1, p2, p3, p4, layout=(2,2), size=(900, 600)))