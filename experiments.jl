"""
experiments.jl — Numerical experiments for "Dynamic Contracts and Corruption"

Generates:
- Table 1 (72 parameter combinations)
- Comparative statics (K* vs κ and ψ)
- Summary statistics for Lemma 1
"""

using DataFrames, CSV, ProgressMeter
include("model.jl")

function run_single_experiment(p::ModelParams; nV=40, nK=40, max_iter=70)
    p = ModelParams(
        sigma=p.sigma, gamma=p.gamma, psi=p.psi, beta=p.beta,
        alpha=p.alpha, delta=p.delta, kappa=p.kappa, tau=p.tau,
        nV=nV, nK=nK, Vmin=p.Vmin, Vmax=p.Vmax, Kmin=p.Kmin, Kmax=p.Kmax,
        max_iter=max_iter, tol=p.tol, feas_tol=p.feas_tol, N_types=p.N_types
    )
    grids = create_grids(p)
    J, policies, stats = solve_model(grids, p; return_stats=true)

    if policies === nothing || !stats.converged
        return (
            sigma = p.sigma, gamma = p.gamma, alpha = p.alpha,
            beta = p.beta, delta = p.delta, tau = p.tau,
            kappa = p.kappa, psi = p.psi,
            V_star = NaN, K_star = NaN,
            rho = NaN, phi = NaN,
            RT = NaN, ET = NaN,
            D_low = NaN,
            corruption_share_low = NaN,
            marginal_return = NaN,
            converged = false,
            iterations = stats.iterations,
            final_diff = stats.final_diff,
            max_violation = stats.max_violation
        )
    end

    V_star, K_star = find_steady_state(J, policies, grids)
    wedges = compute_wedges(V_star, K_star, policies, grids, p)

    return (
        sigma = p.sigma, gamma = p.gamma, alpha = p.alpha,
        beta = p.beta, delta = p.delta, tau = p.tau,
        kappa = p.kappa, psi = p.psi,
        V_star = V_star, K_star = K_star,
        rho = wedges.rho, phi = wedges.phi,
        RT = wedges.RT, ET = wedges.ET,
        D_low = wedges.D[1],
        corruption_share_low = wedges.corruption_share[1],
        marginal_return = wedges.marginal_return,
        converged = stats.converged,
        iterations = stats.iterations,
        final_diff = stats.final_diff,
        max_violation = stats.max_violation
    )
end

function generate_table1(; quick=false)
    println("Generating Table 1...")
    mkpath("output")

    if quick
        param_grid = Dict(
            :sigma => [1.5, 2.0],
            :gamma => [1.2, 1.5],
            :tau   => [0.15, 0.20]
        )
    else
        param_grid = Dict(
            :sigma => [1.5, 2.0, 2.5],
            :gamma => [1.2, 1.5, 2.0],
            :tau   => [0.15, 0.20],
            :kappa => [0.5, 1.0],
            :psi   => [0.8, 1.0]
        )
    end

    combos = vec(collect(Iterators.product(values(param_grid)...)))
    results = Vector{NamedTuple}(undef, length(combos))

    @showprogress for (i, combo) in enumerate(combos)
        p = quick ?
            ModelParams(sigma=combo[1], gamma=combo[2], tau=combo[3]) :
            ModelParams(sigma=combo[1], gamma=combo[2], tau=combo[3], kappa=combo[4], psi=combo[5])
        try
            results[i] = run_single_experiment(p; nV=quick ? 12 : 35, nK=quick ? 12 : 35, max_iter=quick ? 8 : 120)
        catch e
            println("Failed for combo $i: $e")
            results[i] = quick ? (
                sigma=combo[1], gamma=combo[2], alpha=p.alpha,
                beta=p.beta, delta=p.delta, tau=combo[3],
                kappa=p.kappa, psi=p.psi,
                V_star=NaN, K_star=NaN, rho=NaN, phi=NaN,
                RT=NaN, ET=NaN, D_low=NaN, corruption_share_low=NaN,
                marginal_return=NaN, converged=false, iterations=0, final_diff=NaN,
                max_violation=NaN
            ) : (
                sigma=combo[1], gamma=combo[2], alpha=p.alpha,
                beta=p.beta, delta=p.delta, tau=combo[3],
                kappa=combo[4], psi=combo[5],
                V_star=NaN, K_star=NaN, rho=NaN, phi=NaN,
                RT=NaN, ET=NaN, D_low=NaN, corruption_share_low=NaN,
                marginal_return=NaN, converged=false, iterations=0, final_diff=NaN,
                max_violation=NaN
            )
        end
    end

    df = DataFrame(results)
    CSV.write("output/table1_raw.csv", df)

    # Summary
    df_ok = filter(row -> row.converged && isfinite(row.rho) && isfinite(row.D_low), df)
    summary = nrow(df_ok) == 0 ?
        DataFrame(rho=Float64[], mean_D_low=Float64[], std_D_low=Float64[], mean_corruption_share=Float64[]) :
        combine(groupby(df_ok, :rho),
            :D_low => mean => :mean_D_low,
            :D_low => std => :std_D_low,
            :corruption_share_low => mean => :mean_corruption_share
        )
    CSV.write("output/table1_summary.csv", summary)

    println("Table 1 saved with $(nrow(df)) rows ($(nrow(df_ok)) converged).")
    return df
end

function comparative_statics(; quick=false)
    println("Running comparative statics...")
    mkpath("output")

    # vs kappa
    kappa_vals = range(0.2, 3.0, length=quick ? 5 : 12)
    res_k = [run_single_experiment(ModelParams(kappa=k); nV=quick ? 12 : 30, nK=quick ? 12 : 30, max_iter=quick ? 8 : 100) for k in kappa_vals]
    CSV.write("output/comparative_kappa.csv", DataFrame(res_k))

    # vs psi
    psi_vals = range(0.5, 2.0, length=quick ? 5 : 12)
    res_p = [run_single_experiment(ModelParams(psi=ps); nV=quick ? 12 : 30, nK=quick ? 12 : 30, max_iter=quick ? 8 : 100) for ps in psi_vals]
    CSV.write("output/comparative_psi.csv", DataFrame(res_p))

    println("Comparative statics saved.")
end

println("experiments.jl loaded")