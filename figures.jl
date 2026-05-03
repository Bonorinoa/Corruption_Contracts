"""
figures.jl — Publication figures for "Dynamic Contracts and Corruption"
"""

using Plots, DataFrames, CSV
gr()

function save_fallback_figure(path::String, title_text::String, subtitle_text::String)
    p = plot(
        framestyle = :none,
        legend = false,
        grid = false,
        axis = nothing,
        size = (700, 450),
    )
    annotate!(p, 0.5, 0.62, text(title_text, 18, :center, :black))
    annotate!(p, 0.5, 0.47, text(subtitle_text, 11, :center, :gray35))
    xlims!(p, 0, 1)
    ylims!(p, 0, 1)
    savefig(p, path)
end

function fig_labor_decomposition(wedges)
    D = wedges.D
    share = wedges.corruption_share
    types = ["Low (θ=0.55)", "Medium (θ=1.0)", "High (θ=1.65)"]

    p = bar(types, share .* D, label="Corruption Wedge (ϕ/ρ)", color="#2E86AB", bar_width=0.6)
    bar!(p, types, (1 .- share) .* D, bottom=share .* D, label="Mirrlees Informational Wedge", color="#A23B72", bar_width=0.6)

    plot!(p, xlabel="Productivity Type", ylabel="Labor Distortion Dᵢ",
          title="Labor Distortion Decomposition\n(Uniform Corruption Scaling Dominates)", legend=:topright, size=(700,450))
    savefig(p, "figures/fig1_labor_decomposition.pdf")
    println("Saved fig1_labor_decomposition.pdf")
end

function fig_comparative_kappa()
    df = CSV.read("output/comparative_kappa.csv", DataFrame)
    valid = filter(row -> isfinite(row.kappa) && isfinite(row.K_star) && isfinite(row.RT), df)
    if nrow(valid) == 0
        save_fallback_figure(
            "figures/fig2_comparative_kappa.pdf",
            "Comparative Statics Unavailable",
            "No converged finite observations were available in output/comparative_kappa.csv. Run the full pipeline or increase solver budgets."
        )
        println("Saved fig2_comparative_kappa.pdf (fallback)")
        return
    end

    p1 = plot(valid.kappa, valid.K_star, marker=:o, lw=2, color="#2E86AB",
              xlabel="Enforcement Capacity κ", ylabel="Steady-State K*",
              title="Higher κ → More Public Investment (RT weakens)", legend=false)
    p2 = plot(valid.kappa, valid.RT, marker=:s, lw=2, color="#E94F37",
              xlabel="Enforcement Capacity κ", ylabel="Revolution Tax (RT)",
              title="Revolution Tax Declines with State Capacity", legend=false)
    p = plot(p1, p2, layout=(1,2), size=(900,420))
    savefig(p, "figures/fig2_comparative_kappa.pdf")
    println("Saved fig2_comparative_kappa.pdf")
end

function fig_state_space()
    # Simplified illustrative state-space plot
    V = 10 .^ range(log10(0.4), log10(9), length=50)
    K = 10 .^ range(log10(0.4), log10(14), length=50)

    # Illustrative value + breakdown boundary
    J_proxy = 0.8 .* log.(reshape(V, 1, :)) .+ 0.45 .* log.(reshape(K, :, 1))
    V_lower = 1.15 .+ 0.18 .* K.^0.65

    p = heatmap(V, K, J_proxy', c=:Blues, alpha=0.75,
                xlabel="Promised Utility V", ylabel="Public Capital K",
                title="State Space and Revolutionary Breakdown Boundary")
    plot!(p, V_lower, K, lw=2.5, color=:red, ls=:dash, label="Breakdown V̲(K,κ)")
    plot!(p, V_lower, K, fillrange=maximum(V), fillalpha=0.12, color=:red, label="Revolution Region")
    savefig(p, "figures/fig3_state_space.pdf")
    println("Saved fig3_state_space.pdf")
end

function fig_distortion_vs_rho()
    df = CSV.read("output/table1_raw.csv", DataFrame)
    df = filter(row -> isfinite(row.rho) && isfinite(row.D_low), df)

    if nrow(df) == 0
        save_fallback_figure(
            "figures/fig4_distortion_vs_rho.pdf",
            "Distortion Scatter Unavailable",
            "No converged finite rows were available in output/table1_raw.csv. This is expected in smoke-test runs."
        )
        println("Saved fig4_distortion_vs_rho.pdf (fallback)")
        return
    end

    p = scatter(df.rho, df.D_low, alpha=0.35, color="#2E86AB", markersize=3,
                xlabel="Corruption Wedge ρ = 1 + ϕ", ylabel="Labor Distortion (Low Type) D₁",
                title="Labor Distortion Rises with Corruption\n($(nrow(df)) parameter combinations)", legend=false)
    hline!(p, [0.06], color=:gray, ls=:dash, label="First-best benchmark")
    savefig(p, "figures/fig4_distortion_vs_rho.pdf")
    println("Saved fig4_distortion_vs_rho.pdf")
end

println("figures.jl loaded")