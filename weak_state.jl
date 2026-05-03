using JuMP
using Ipopt
using Plots

# =====================================================================
# 1. PARAMETERS
# =====================================================================
const β = 0.95      
const σ = 2.0       
const γ = 2.0       
const ψ = 1.0       
const α = 0.3       
const δ = 0.1       

const θ = [0.8, 1.2]      
const π_prob = [0.5, 0.5] 

const T = 30        
const K1 = 0.5      
const V_rev = -6.0  

println("Solving the Decentralized Weak State (Ramsey Approach)...")

m_weak = Model(Ipopt.Optimizer)
set_silent(m_weak)

# Variables
@variable(m_weak, c_w[1:2, 1:T] >= 0.01)
@variable(m_weak, 0.01 <= l_w[1:2, 1:T] <= 0.99)
@variable(m_weak, K_w[1:T+1] >= 0.01)
@variable(m_weak, S_w[1:T] >= 0.0)
@variable(m_weak, 0.0 <= tau[1:T] <= 0.99) # Linear expropriation tax

# Initial Capital
@constraint(m_weak, K_w[1] == K1)

# Objective: Maximize Stolen Rents
@objective(m_weak, Max, sum( β^(t-1) * S_w[t] for t in 1:T ))

for t in 1:T
    # 1. Government Budget/Resource Constraint
    # Total output Y_t
    Y_t = sum(π_prob[i] * (K_w[t]^α) * ((θ[i] * (1 - l_w[i,t]))^(1 - α)) for i in 1:2)
    # The government collects tau*Y_t, invests in K_{t+1}, and steals S_w
    @constraint(m_weak, K_w[t+1] - (1-δ)*K_w[t] + S_w[t] == tau[t] * Y_t)
    
    for i in 1:2
        # 2. Citizen's Decentralized Budget Constraint
        # Citizens consume exactly what they keep after the expropriation tax
        @constraint(m_weak, c_w[i,t] == (1 - tau[t]) * (K_w[t]^α) * ((θ[i] * (1 - l_w[i,t]))^(1 - α)))
        
        # 3. Citizen's Optimal Labor Choice (The Implementability Constraint)
        # Marginal Utility of Leisure == Marginal Utility of Consumption * Net Marginal Product of Labor
        @constraint(m_weak, 
            ψ * l_w[i,t]^(-γ) == 
            c_w[i,t]^(-σ) * (1 - tau[t]) * (1 - α) * (K_w[t]^α) * (θ[i]^(1 - α)) * ((1 - l_w[i,t])^(-α))
        )
    end
    
    # 4. Threat of Revolution Constraint
    @constraint(m_weak, sum(π_prob[i] * ( (c_w[i,t]^(1-σ))/(1-σ) + ψ*(l_w[i,t]^(1-γ))/(1-γ) ) for i in 1:2) >= V_rev)
end

optimize!(m_weak)

K_weak_path = value.(K_w)[1:T]
l_weak_low_path = value.(l_w)[1, :]
tau_path = value.(tau)

# Plotting
p1 = plot(1:T, K_weak_path, label="Weak State Capital", lw=3, color=:purple, title="Public Capital", xlabel="Time")
p2 = plot(1:T, l_weak_low_path, label="Informality (Low Type)", lw=3, color=:orange, title="Informal Time (Leisure)", xlabel="Time")
p3 = plot(1:T, tau_path, label="Expropriation Rate (tau)", lw=3, color=:red, title="Govt Tax Rate", xlabel="Time")
display(plot(p1, p2, p3, layout=(1,3), size=(1200, 400)))