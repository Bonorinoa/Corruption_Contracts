"""
model.jl — Core recursive model for "Dynamic Contracts and Corruption"

This implements:
- Primitives (u, v, F, Vrev, Jdev)
- Discretized state space (V, K) with log-spacing
- Bellman operator using NLopt (LN_AUGLAG + LD_SLSQP) — robust for non-convex GIC
- Value function iteration with warm-start policy
- Steady-state finder
- Wedge calculator (RT, ET, labor distortion decomposition)

Usage:
    include("model.jl")
    params = Baseline()
    grids = create_grids(params)
    J, policies = solve_model(grids, params)
    V_star, K_star = find_steady_state(J, policies, grids)
    wedges = compute_wedges(V_star, K_star, policies, grids, params)
"""

using NLopt
using Interpolations
using Statistics
using LinearAlgebra

# ============================================================
# 1. PARAMETER STRUCT (clean, immutable)
# ============================================================
Base.@kwdef struct ModelParams
    # Preferences
    sigma::Float64 = 2.0          # CRRA
    gamma::Float64 = 1.5          # Inverse Frisch for informal labor
    psi::Float64   = 1.0          # Scale of informal disutility
    beta::Float64  = 0.96         # Discount factor

    # Technology
    alpha::Float64 = 0.33         # Capital share
    delta::Float64 = 0.10         # Depreciation

    # Political economy
    kappa::Float64 = 1.0          # Enforcement capacity
    tau::Float64   = 0.20         # Revolution destruction fraction

    # Grids
    nV::Int = 60
    nK::Int = 60
    Vmin::Float64 = 0.4
    Vmax::Float64 = 9.0
    Kmin::Float64 = 0.4
    Kmax::Float64 = 14.0

    # Optimization
    max_iter::Int = 200
    tol::Float64  = 1e-3
    feas_tol::Float64 = 1e-4
    N_types::Int  = 3
end

Baseline() = ModelParams()

# ============================================================
# 2. PRIMITIVES
# ============================================================
u(c; sigma=2.0) = sigma == 1 ? log(max(c, 1e-10)) : (max(c, 1e-10)^(1-sigma) - 1) / (1 - sigma)
v(l; psi=1.0, gamma=1.5) = psi * (max(l, 1e-10)^(1-gamma) - 1) / (1 - gamma)

F(K, theta, e; alpha=0.33) = K^alpha * (theta * e)^(1-alpha)
Fe(K, theta, e; alpha=0.33) = (1-alpha) * K^alpha * theta^(1-alpha) * e^(-alpha)

function Vrev(K; kappa=1.0, tau=0.20)
    V_eff = 0.75 * K^0.42          # placeholder for post-revolution value
    h = min(1.0, 0.6 * kappa * K / (1 + kappa * K))
    return (1 - h) * V_eff * (1 - tau)
end

Jdev(Kp; delta=0.10) = (1 - delta) * Kp

# ============================================================
# 3. GRIDS
# ============================================================
function create_grids(p::ModelParams = Baseline())
    V = 10 .^ range(log10(p.Vmin), log10(p.Vmax), length=p.nV)
    K = 10 .^ range(log10(p.Kmin), log10(p.Kmax), length=p.nK)

    if p.N_types == 3
        theta = [0.55, 1.00, 1.65]
        pi    = [0.30, 0.40, 0.30]
    else
        theta = [0.7, 1.0, 1.4, 1.8][1:p.N_types]
        pi    = fill(1/p.N_types, p.N_types)
    end
    pi ./= sum(pi)

    return (V=V, K=K, theta=theta, pi=pi)
end

# ============================================================
# 4. BELLMAN OPERATOR (NLopt — robust)
# ============================================================
function bellman_operator(J_old::Matrix{Float64}, grids, p::ModelParams; init_policies=nothing)
    V, K, theta, pi = grids.V, grids.K, grids.theta, grids.pi
    nV, nK, N = length(V), length(K), length(theta)

    J_new = zeros(nV, nK)
    policies = (
        c  = zeros(nV, nK, N),
        l  = zeros(nV, nK, N),
        W  = zeros(nV, nK, N),
        Kp = zeros(nV, nK),
        S  = zeros(nV, nK),
        phi = zeros(nV, nK),
        xi  = zeros(nV, nK, N)
    )

    # Interpolator for continuation value
    itp = LinearInterpolation((V, K), J_old, extrapolation_bc=Flat())

    for iV in 1:nV, iK in 1:nK
        v_val, k_val = V[iV], K[iK]

        # Warm start from previous-iteration policies when available.
        if init_policies !== nothing
            c0 = vec(init_policies.c[iV, iK, :])
            l0 = vec(init_policies.l[iV, iK, :])
            W0 = vec(init_policies.W[iV, iK, :])
            Kp0 = init_policies.Kp[iV, iK]
            S0 = init_policies.S[iV, iK]
            if !all(isfinite.(c0)) || !all(isfinite.(l0)) || !all(isfinite.(W0)) || !isfinite(Kp0) || !isfinite(S0)
                c0 = fill(0.65 * k_val, N)
                l0 = fill(0.22, N)
                W0 = fill(max(0.3, v_val * 0.95), N)
                Kp0 = 0.88 * k_val
                S0 = 0.03 * k_val
            end
        else
            c0 = fill(0.65 * k_val, N)
            l0 = fill(0.22, N)
            W0 = fill(max(0.3, v_val * 0.95), N)
            Kp0 = 0.88 * k_val
            S0 = 0.03 * k_val
        end
        x0 = vcat(c0, l0, W0, Kp0, S0)

        # Bounds (finite)
        c_max  = 2.8 * k_val + 1.5
        l_max  = 0.98
        W_max  = V[end] + 1.5
        Kp_max = K[end] * 1.4
        S_max  = 1.8 * k_val + 0.5

        lb = vcat(fill(1e-4, 2N), fill(max(0.1, V[1]-0.5), N), K[1]/2, 0.0)
        ub = vcat(fill(c_max, N), fill(l_max, N), fill(W_max, N), Kp_max, S_max)

        function objective(x::Vector, grad::Vector)
            if length(grad) > 0
                grad .= 0.0
            end
            c, l, W, Kp, S = x[1:N], x[N+1:2N], x[2N+1:3N], x[3N+1], x[3N+2]
            Wbar = sum(pi .* W)
            val = S + p.beta * itp(Wbar, Kp)
            return isfinite(val) ? val : -1e12
        end

        function gic(x::Vector, grad::Vector)
            if length(grad) > 0; grad .= 0.0; end
            c, l, W, Kp, S = x[1:N], x[N+1:2N], x[2N+1:3N], x[3N+1], x[3N+2]
            Wbar = sum(pi .* W)
            val = S + p.beta * itp(Wbar, Kp) - Jdev(Kp; delta=p.delta)
            return isfinite(val) ? val : -1e6
        end

        function pk_fun(x::Vector, grad::Vector)
            if length(grad) > 0; grad .= 0.0; end
            c, l, W = x[1:N], x[N+1:2N], x[2N+1:3N]
            return v_val - sum(pi .* (u.(c; sigma=p.sigma) .+ v.(l; psi=p.psi, gamma=p.gamma) .+ p.beta .* W))
        end

        function rc_fun(x::Vector, grad::Vector)
            if length(grad) > 0; grad .= 0.0; end
            c, l, W, Kp, S = x[1:N], x[N+1:2N], x[2N+1:3N], x[3N+1], x[3N+2]
            y = [F(k_val, theta[j], 1-l[j]; alpha=p.alpha) for j in 1:N]
            return sum(pi .* c) + Kp + S - (sum(pi .* y) + (1-p.delta)*k_val)
        end

        function ic_max_violation(x::Vector)
            c, l, W = x[1:N], x[N+1:2N], x[2N+1:3N]
            worst = -Inf
            for j in 2:N
                e_m = theta[j-1] / theta[j] * (1 - l[j-1])
                l_m = 1 - e_m
                # mimic utility - truthful utility <= 0
                v_ic = (u(c[j-1]; sigma=p.sigma) + v(l_m; psi=p.psi, gamma=p.gamma) + p.beta * W[j-1]) -
                       (u(c[j]; sigma=p.sigma) + v(l[j]; psi=p.psi, gamma=p.gamma) + p.beta * W[j])
                worst = max(worst, v_ic)
            end
            return worst
        end

        function pc_violation(x::Vector)
            c, l, W = x[1:N], x[N+1:2N], x[2N+1:3N]
            vrev = Vrev(k_val; kappa=p.kappa, tau=p.tau)
            return vrev - minimum(u.(c; sigma=p.sigma) .+ v.(l; psi=p.psi, gamma=p.gamma) .+ p.beta .* W)
        end

        function gic_violation(x::Vector)
            return -gic(x, Float64[])
        end

        function max_constraint_violation(x::Vector)
            return maximum([
                abs(pk_fun(x, Float64[])),
                abs(rc_fun(x, Float64[])),
                ic_max_violation(x),
                pc_violation(x),
                gic_violation(x)
            ])
        end

        # Constraints
        cons = []
        # PK and RC are equalities -> enforce both signs in c(x) <= 0 form.
        push!(cons, pk_fun)
        push!(cons, (x, grad) -> -pk_fun(x, grad))
        push!(cons, rc_fun)
        push!(cons, (x, grad) -> -rc_fun(x, grad))
        # IC (downward only)
        for j in 2:N
            push!(cons, (x, grad) -> begin
                if length(grad) > 0; grad .= 0.0; end
                c, l, W = x[1:N], x[N+1:2N], x[2N+1:3N]
                e_m = theta[j-1]/theta[j] * (1 - l[j-1])
                l_m = 1 - e_m
                (u(c[j-1]; sigma=p.sigma) + v(l_m; psi=p.psi, gamma=p.gamma) + p.beta*W[j-1]) -
                (u(c[j]; sigma=p.sigma) + v(l[j]; psi=p.psi, gamma=p.gamma) + p.beta*W[j])
            end)
        end
        # PC
        push!(cons, (x, grad) -> begin
            if length(grad) > 0; grad .= 0.0; end
            c, l, W = x[1:N], x[N+1:2N], x[2N+1:3N]
            vrev = Vrev(k_val; kappa=p.kappa, tau=p.tau)
            vrev - minimum(u.(c; sigma=p.sigma) .+ v.(l; psi=p.psi, gamma=p.gamma) .+ p.beta .* W)
        end)
        # GIC
        push!(cons, (x, grad) -> -gic(x, grad))

        # Keep initial guess strictly within bounds.
        x0 = clamp.(x0, lb .+ 1e-8, ub .- 1e-8)

        function run_optimizer(alg::Symbol, x_init::Vector{Float64}; use_local::Bool=false)
            opt = Opt(alg, length(x_init))
            opt.lower_bounds = lb
            opt.upper_bounds = ub
            opt.max_objective = objective
            for c in cons
                inequality_constraint!(opt, c, 1e-6)
            end
            opt.xtol_rel = 1e-6
            opt.ftol_rel = 1e-8
            opt.maxeval = 2500
            opt.initial_step = 0.05

            if use_local
                local_opt = Opt(:LN_COBYLA, length(x_init))
                local_opt.xtol_rel = 1e-8
                local_opt.ftol_rel = 1e-8
                local_opt.maxeval = 1500
                opt.local_optimizer = local_opt
            end

            return optimize(opt, x_init)
        end

        fval = -1e12
        minx = copy(x0)

        try
            fval, minx, _ = run_optimizer(:LN_AUGLAG, x0; use_local=true)
        catch
            # Fall back to COBYLA if AUGLAG fails hard.
            try
                fval, minx, _ = run_optimizer(:LN_COBYLA, x0)
            catch
                fval = -1e12
                minx .= x0
            end
        end

        # If constraints are still poor, run a direct feasibility-oriented pass.
        if !isfinite(fval) || max_constraint_violation(minx) > 1e-4
            try
                fval2, minx2, _ = run_optimizer(:LN_COBYLA, minx)
                if isfinite(fval2) && max_constraint_violation(minx2) <= max_constraint_violation(minx)
                    fval = fval2
                    minx = minx2
                end
            catch
                # Keep best available iterate.
            end
        end

        # Final rescue: try a few focused restarts, prioritize lower violation.
        if !isfinite(fval) || max_constraint_violation(minx) > 5e-4
            candidates = Vector{Tuple{Float64, Vector{Float64}}}()
            push!(candidates, (fval, copy(minx)))

            for (alg, seed) in ((:LN_AUGLAG, x0), (:LN_AUGLAG, minx), (:LN_COBYLA, x0), (:LN_COBYLA, minx))
                try
                    fv_try, x_try, _ = run_optimizer(alg, copy(seed); use_local=(alg == :LN_AUGLAG))
                    if isfinite(fv_try)
                        push!(candidates, (fv_try, copy(x_try)))
                    end
                catch
                    # Ignore failed restart.
                end
            end

            best_idx = 1
            best_v = max_constraint_violation(candidates[1][2])
            for idx in 2:length(candidates)
                v_now = max_constraint_violation(candidates[idx][2])
                if (v_now < best_v - 1e-8) || (abs(v_now - best_v) <= 1e-8 && candidates[idx][1] > candidates[best_idx][1])
                    best_idx = idx
                    best_v = v_now
                end
            end

            fval = candidates[best_idx][1]
            minx = candidates[best_idx][2]
        end

        # Store results
        c_opt, l_opt, W_opt, Kp_opt, S_opt = minx[1:N], minx[N+1:2N], minx[2N+1:3N], minx[3N+1], minx[3N+2]
        J_new[iV, iK] = isfinite(fval) ? fval : NaN
        policies.c[iV, iK, :] .= c_opt
        policies.l[iV, iK, :] .= l_opt
        policies.W[iV, iK, :] .= W_opt
        policies.Kp[iV, iK] = Kp_opt
        policies.S[iV, iK] = S_opt
        policies.phi[iV, iK] = max(0.0, -gic(minx, Float64[]))
        policies.xi[iV, iK, :] .= pi
    end

    return J_new, policies
end

# ============================================================
# 5. VALUE FUNCTION ITERATION
# ============================================================
function policy_max_violation(J::Matrix{Float64}, policies, grids, p::ModelParams)
    if policies === nothing
        return Inf
    end

    V, K, theta, pi = grids.V, grids.K, grids.theta, grids.pi
    N = length(theta)
    itp = LinearInterpolation((V, K), J, extrapolation_bc=Flat())

    max_v = 0.0
    for iV in eachindex(V), iK in eachindex(K)
        v_val, k_val = V[iV], K[iK]
        c = vec(policies.c[iV, iK, :])
        l = vec(policies.l[iV, iK, :])
        W = vec(policies.W[iV, iK, :])
        Kp = policies.Kp[iV, iK]
        S = policies.S[iV, iK]

        if !all(isfinite.(c)) || !all(isfinite.(l)) || !all(isfinite.(W)) || !isfinite(Kp) || !isfinite(S)
            return Inf
        end

        truthful_u = u.(c; sigma=p.sigma) .+ v.(l; psi=p.psi, gamma=p.gamma) .+ p.beta .* W
        pk_eq = abs(v_val - sum(pi .* truthful_u))

        y = [F(k_val, theta[j], 1-l[j]; alpha=p.alpha) for j in 1:N]
        rc_eq = abs(sum(pi .* c) + Kp + S - (sum(pi .* y) + (1-p.delta)*k_val))

        ic_viol = 0.0
        for j in 2:N
            e_m = theta[j-1] / theta[j] * (1 - l[j-1])
            l_m = 1 - e_m
            v_ic = (u(c[j-1]; sigma=p.sigma) + v(l_m; psi=p.psi, gamma=p.gamma) + p.beta * W[j-1]) -
                   (u(c[j]; sigma=p.sigma) + v(l[j]; psi=p.psi, gamma=p.gamma) + p.beta * W[j])
            ic_viol = max(ic_viol, v_ic)
        end

        pc_viol = Vrev(k_val; kappa=p.kappa, tau=p.tau) - minimum(truthful_u)
        Wbar = sum(pi .* W)
        gic_viol = Jdev(Kp; delta=p.delta) - (S + p.beta * itp(Wbar, Kp))

        local_max = maximum([pk_eq, rc_eq, max(0.0, ic_viol), max(0.0, pc_viol), max(0.0, gic_viol)])
        max_v = max(max_v, local_max)
    end
    return max_v
end

function solve_model(grids, p::ModelParams = Baseline(); return_stats::Bool=false)
    V, K = grids.V, grids.K
    J = zeros(length(V), length(K))
    policies = nothing
    damping = 0.6
    converged = false
    final_diff = Inf
    max_violation = Inf
    completed_iters = 0

    for it in 1:p.max_iter
        J_new_raw, policies = bellman_operator(J, grids, p; init_policies=policies)
        J_new = damping .* J_new_raw .+ (1 - damping) .* J
        diff = maximum(abs.(J_new .- J))
        completed_iters = it
        final_diff = diff
        if !isfinite(diff)
            println("NaN/Inf detected at iteration $it — stopping early")
            stats = (converged=false, iterations=it, final_diff=diff, max_violation=Inf)
            return return_stats ? (J, policies, stats) : (J, policies)
        end
        J .= J_new
        println("Iter $it | diff = $(round(diff, digits=8))")
        if diff < p.tol
            max_violation = policy_max_violation(J, policies, grids, p)
            if isfinite(max_violation) && max_violation <= p.feas_tol
                println("Converged after $it iterations (max violation = $(round(max_violation, digits=8)))")
                converged = true
                stats = (converged=true, iterations=it, final_diff=diff, max_violation=max_violation)
                return return_stats ? (J, policies, stats) : (J, policies)
            else
                println("Bellman diff below tol but constraints still loose (max violation = $(round(max_violation, digits=8)))")
            end
        end
    end
    max_violation = policy_max_violation(J, policies, grids, p)
    println("Did not fully converge (final diff = $(final_diff), max violation = $(max_violation))")
    stats = (converged=converged, iterations=completed_iters, final_diff=final_diff, max_violation=max_violation)
    return return_stats ? (J, policies, stats) : (J, policies)
end

# ============================================================
# 6. STEADY STATE
# ============================================================
function find_steady_state(J, policies, grids)
    V, K, pi = grids.V, grids.K, grids.pi
    Wbar = dropdims(sum(policies.W .* reshape(pi, 1, 1, :), dims=3), dims=3)
    dist = abs.(Wbar .- V) .+ abs.(policies.Kp .- reshape(K, 1, :))
    idx = argmin(dist)
    return V[idx[1]], K[idx[2]]
end

# ============================================================
# 7. WEDGE CALCULATOR
# ============================================================
function compute_wedges(V_star, K_star, policies, grids, p::ModelParams)
    V, K, theta, pi = grids.V, grids.K, grids.theta, grids.pi
    iV = argmin(abs.(V .- V_star))
    iK = argmin(abs.(K .- K_star))

    c = policies.c[iV, iK, :]
    l = policies.l[iV, iK, :]
    phi = policies.phi[iV, iK]
    rho = 1 + phi

    # Marginal return
    FK = [p.alpha * F(K_star, theta[j], 1-l[j]; alpha=p.alpha) / K_star for j in 1:length(theta)]
    marg_ret = sum(pi .* FK) + (1 - p.delta)

    # Revolution Tax (finite difference)
    dVrev = (Vrev(K_star*1.01; kappa=p.kappa, tau=p.tau) - Vrev(K_star*0.99; kappa=p.kappa, tau=p.tau)) / (0.02*K_star)
    RT = sum(policies.xi[iV, iK, :] .* dVrev) / rho

    # Expropriation Tax
    ET = phi * (1 - p.delta) / rho

    # Labor distortions
    D = [1 - v(l[j]; psi=p.psi, gamma=p.gamma) / (u(c[j]; sigma=p.sigma) * Fe(K_star, theta[j], 1-l[j]; alpha=p.alpha)) for j in 1:length(theta)]
    corruption_share = (phi / rho) ./ (D .+ 1e-8)

    return (rho=rho, RT=RT, ET=ET, D=D, corruption_share=corruption_share,
            marginal_return=marg_ret, c=c, l=l, phi=phi)
end

println("model.jl loaded successfully")