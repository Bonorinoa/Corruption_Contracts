using JuMP
using Ipopt
using Plots

# =====================================================================
# 1. PARAMETERS 
# =====================================================================
const β = 0.95      # Discount factor
const σ = 2.0       # Risk aversion
const γ = 2.0       # Inverse Frisch elasticity
const ψ = 1.0       # Weight on informal/home production
const α = 0.3       # Output elasticity of public capital
const δ = 0.1       # Depreciation rate

const θ = [0.8, 1.2]      # Low and High type productivity
const π_prob = [0.5, 0.5] # 50% chance of each

const T = 30        # Time horizon (periods)
const K1 = 0.5      # Initial capital stock (Start poor)
const V_rev = -6.0  # Per-period threat of revolution

println("Solving Dynamic Transition Paths (T = $T)...")

# =====================================================================
# MODEL 1: DYNAMIC FIRST-BEST (Benevolent Planner)
# =====================================================================
m_fb = Model(Ipopt.Optimizer)
set_silent(m_fb)

@variable(m_fb, c[1:2, 1:T] >= 0.01)
@variable(m_fb, 0.01 <= l[1:2, 1:T] <= 0.99)
@variable(m_fb, K[1:T+1] >= 0.01)

# Initial Capital Condition
@constraint(m_fb, K[1] == K1)

# Objective: Maximize Present Value of Expected Utility
@objective(m_fb, Max, sum( β^(t-1) * sum(π_prob[i] * ((c[i,t]^(1-σ))/(1-σ) + ψ*(l[i,t]^(1-γ))/(1-γ)) for i in 1:2) for t in 1:T ))

# Dynamic Resource Constraint
for t in 1:T
    @constraint(m_fb, sum(π_prob[i] * c[i,t] for i in 1:2) + K[t+1] == sum(π_prob[i] * (K[t]^α) * ((θ[i] * (1 - l[i,t]))^(1 - α)) for i in 1:2) + (1-δ)*K[t])
end

optimize!(m_fb)
K_fb_path = value.(K)[1:T]
l_fb_low_path = value.(l)[1, :]

# =====================================================================
# MODEL 2: DYNAMIC CORRUPTION CONTRACT (Strong State)
# =====================================================================
m_corr = Model(Ipopt.Optimizer)
set_silent(m_corr)

@variable(m_corr, c_c[1:2, 1:T] >= 0.01)
@variable(m_corr, 0.01 <= l_c[1:2, 1:T] <= 0.99)
@variable(m_corr, K_c[1:T+1] >= 0.01)
@variable(m_corr, S[1:T] >= 0.0)

# Initial Capital Condition
@constraint(m_corr, K_c[1] == K1)

# Objective: Maximize Present Value of Stolen Rents
@objective(m_corr, Max, sum( β^(t-1) * S[t] for t in 1:T ))

for t in 1:T
    # Dynamic Resource Constraint (with Stolen Rents S_t)
    @constraint(m_corr, sum(π_prob[i] * c_c[i,t] for i in 1:2) + K_c[t+1] + S[t] == sum(π_prob[i] * (K_c[t]^α) * ((θ[i] * (1 - l_c[i,t]))^(1 - α)) for i in 1:2) + (1-δ)*K_c[t])
    
    # Incentive Compatibility Constraint (High mimicking Low)
    @constraint(m_corr, 
        ((c_c[2,t]^(1-σ))/(1-σ) + ψ*(l_c[2,t]^(1-γ))/(1-γ)) >= 
        ((c_c[1,t]^(1-σ))/(1-σ) + ψ*( (1.0 - (θ[1]/θ[2])*(1.0 - l_c[1,t]))^(1-γ) )/(1-γ))
    )
    
    # Threat of Revolution Constraint (Per-period utility must be >= V_rev)
    @constraint(m_corr, sum(π_prob[i] * ( (c_c[i,t]^(1-σ))/(1-σ) + ψ*(l_c[i,t]^(1-γ))/(1-γ) ) for i in 1:2) >= V_rev)
end

optimize!(m_corr)
K_corr_path = value.(K_c)[1:T]
l_corr_low_path = value.(l_c)[1, :]
S_path = value.(S)

# =====================================================================
# VISUALIZATION (Phase Diagrams)
# =====================================================================
println("Optimization Complete! Generating phase diagrams...")

# 1. Capital Accumulation Path
p1 = plot(1:T, K_fb_path, label="First-Best", linewidth=3, color=:blue, 
          title="Public Capital (K)", xlabel="Time (t)", ylabel="Capital Stock", legend=:bottomright)
plot!(p1, 1:T, K_corr_path, label="Corruption", linewidth=3, color=:red, linestyle=:dash)

# 2. Extracted Rents Path
p2 = plot(1:T, S_path, label="Stolen Rents (S)", linewidth=3, color=:green, fill=(0, 0.2, :green),
          title="Govt Expropriation", xlabel="Time (t)", ylabel="Stolen Output", legend=:topright)

# 3. Informal Time / Leisure Path (Low Type)
p3 = plot(1:T, l_fb_low_path, label="First-Best", linewidth=3, color=:blue, 
          title="Informal Time (Low Type)", xlabel="Time (t)", ylabel="Leisure / Informality", legend=:right)
plot!(p3, 1:T, l_corr_low_path, label="Corruption", linewidth=3, color=:red, linestyle=:dash)

display(plot(p1, p2, p3, layout=(1,3), size=(1200, 400), margin=5Plots.mm))