"""
run_all.jl — Master replication script (Julia version)
"""

include("model.jl")
include("experiments.jl")
include("figures.jl")

function main(; quick::Bool = false)
    println("="^60)
    println("DYNAMIC CONTRACTS AND CORRUPTION — JULIA REPLICATION")
    println("="^60)

    mkpath("output"); mkpath("figures")

    # 1. Baseline
    println("\n[1/4] Solving baseline...")
    p = quick ? ModelParams(nV=12, nK=12, max_iter=12) : Baseline()
    grids = create_grids(p)
    J, policies, stats = solve_model(grids, p; return_stats=true)

    if policies === nothing
        error("Baseline solver failed before producing policy functions.")
    end

    if !stats.converged
        msg = "Baseline solver did not converge (iterations=$(stats.iterations), final_diff=$(round(stats.final_diff, digits=6)), max_violation=$(round(stats.max_violation, digits=6)))."
        if quick
            println("WARNING: " * msg)
        else
            error(msg * " Aborting full pipeline to avoid publishing unreliable results.")
        end
    end

    V_star, K_star = find_steady_state(J, policies, grids)
    wedges = compute_wedges(V_star, K_star, policies, grids, p)

    println("Baseline: K* = $(round(K_star, digits=3)) | ρ = $(round(wedges.rho, digits=3)) | D₁ = $(round(wedges.D[1], digits=4))")
    println("RT = $(round(wedges.RT, digits=4)) | ET = $(round(wedges.ET, digits=4)) | converged = $(stats.converged) | max_violation = $(round(stats.max_violation, digits=6))")

    # 2. Experiments
    if !quick
        println("\n[2/4] Running full experiment grid...")
        generate_table1()
        comparative_statics()
    else
        println("\n[2/4] Quick mode — running reduced experiments for smoke testing")
        generate_table1(quick=true)
        comparative_statics(quick=true)
    end

    # 3. Figures
    println("\n[3/4] Generating figures...")
    fig_labor_decomposition(wedges)
    fig_comparative_kappa()
    fig_state_space()
    fig_distortion_vs_rho()

    println("\n" * ("="^60))
    println("REPLICATION COMPLETE")
    println("="^60)
    println("Figures saved to: figures/")
    println("Data saved to:    output/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(quick=false)
end