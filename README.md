# Replication Code (Julia) — Dynamic Contracts and Corruption

**Primary implementation**: Julia 1.9+  
**Paper**: Gonzalez-Bonorino (2026), "Dynamic Contracts and Corruption: A Recursive Macroeconomic Approach"

## Quick Start

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("run_all.jl")
main(quick=true)     # baseline + figures only (~3-5 min)
# main(quick=false)  # full 72-combination grid (~40-70 min)
```

## Files

| File            | Description |
|-----------------|-------------|
| `Project.toml`  | Julia environment |
| `model.jl`      | Primitives, NLopt-based Bellman operator, VFI, wedges |
| `experiments.jl`| Table 1 (72 combos), comparative statics |
| `figures.jl`    | 4 publication-quality figures |
| `run_all.jl`    | Master script |

## Key Numerical Results (reproduced)

- Labor distortion $D_1$ strictly increasing in corruption wedge $\rho = 1 + \phi$ for > 92% of the parameter space.
- Corruption (uniform scaling) accounts for 68–93% of total labor distortion.
- Both Revolution Tax (RT) and Expropriation Tax (ET) strictly positive → chronic under-investment.
- $\partial K^*/\partial \kappa > 0$ (enforcement capacity raises public capital).

## Notes on Implementation

- Uses `NLopt` with `LN_AUGLAG` (augmented Lagrangian) + `LD_SLSQP` local solver — far more robust than pure SLSQP for the non-convex GIC.
- Warm-start from previous policy greatly improves convergence speed.
- Log-spaced grids + linear interpolation.
- Full 60×60 grids used for production runs.

## Python Reference Version (Archived)

A Python prototype was used during development to validate the theoretical structure. The Julia version is the production implementation for the paper and replication package.
| `figures.py`      | Labor decomposition, K* vs κ, state space    | ~30 sec        |
| `run_all.py`      | Master script (calls all above)              | full / quick   |

## Key Numerical Results Reproduced

- **Lemma 1 (Approximate Separation)**: Labor distortion $D_i$ increasing in corruption wedge $\rho = 1 + \phi$ for >90% of parameter space. Corruption (uniform scaling) accounts for 63–94% of total distortion.
- **Two Political Wedges**: Both Revolution Tax (RT) and Expropriation Tax (ET) strictly positive at baseline → chronic under-investment ($K^* < K^{FB}$).
- **Inverse Euler survives**: Confirmed numerically (expectation form holds after political frictions).
- **Comparative Statics**: $\partial K^*/\partial \kappa > 0$ (enforcement capacity raises public capital); $\partial K^*/\partial \psi$ ambiguous (as predicted).

## Notes on Implementation

- **Non-convexity**: GIC handled via constrained NLP (`scipy.optimize.SLSQP` + differential evolution fallback) at each state. Full paper uses lottery extensions / dual-recursive formulation for rigor.
- **Performance**: Python prototype uses coarse grids (25×25) and N=3 types. Production version (Julia) uses 80×80 grids + parallel VFI + NLopt.
- **Single Crossing**: Verified numerically in `model.py` for baseline params (Assumption 1 holds).
- **Breakdown Boundary**: `V̲(K, κ)` characterized numerically (Section 3.1); plotted in Figure 3.

## Porting to Julia (Recommended for Full Paper)

The Python version is for rapid prototyping and review. For the final JEDC submission:

1. Port `model.py` → Julia (use `QuantEcon.jl`, `Optim.jl`, `Interpolations.jl`).
2. Replace grid loops with `@threads` or `Distributed`.
3. Add collocation / Chebyshev polynomials for smoother policies.
4. Full 72× finer grid + calibration to cross-country data (World Bank, ICRG corruption index, ILO informal employment).

See companion repository branch `julia-version` (to be added).

## Contact / Issues

Augusto Gonzalez-Bonorino  
agbonori@asu.edu  
GitHub: https://github.com/Bonorinoa/Corruption_Contracts

---

**Citation**  
Gonzalez-Bonorino, A. (2026). "Dynamic Contracts and Corruption: A Recursive Macroeconomic Approach." Working Paper, Arizona State University.