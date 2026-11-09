# PlasmaPopulations.jl

Shared vocabulary for describing plasma: particles, velocity distribution models and populations.

```julia
using PlasmaPopulations, Unitful

p, e = Proton(), Electron()
plasma = Plasma(
    Population(p, BiMaxwellian; T_perp = 100u"eV", T_para = 20u"eV", n = 5u"cm^-3"),
    Population(e, BiMaxwellian; T = 50u"eV", n = 5u"cm^-3");
    B0 = 5u"nT",
)
PlasmaPopulations.alfven_speed(plasma)   # m/s

rand(BiMaxwellian(p; T_perp = 100u"eV", T_para = 20u"eV"), 1000)   # field-aligned velocities, m/s
```

## Conventions

- Bare numbers (including inputs, stored fields and formulary outputs) are assumed in SI and are stored as given, except temperature/energy in eV.
  Unitful inputs are converted to SI at construction.
- Velocities are momenta per mass `𝐮 = γ𝐯 = 𝐩/m`.
- `B0` is a magnitude or a 3-vector. Velocities and wavevectors are in the frame with `ẑ` along the field; the models here are gyrotropic about it.
  `Oriented` places a model in another frame.
- Thermal speeds are moments for every model: `vth_perp² = ⟨v⊥²⟩` and `vth_para² = 2⟨(v∥ - vd_para)²⟩`, so `T = m vth²/2`.
- Constructors take keywords named after the fields, plus `vth` for `vth_perp = vth_para`.
  `V(particle; kw...)` also accepts `T_perp`, `T_para` or `T` (both) in place of `vth_perp`, `vth_para`, for models with those parameters.

## API

| name | meaning |
|---|---|
| `Particle(q, m)` | charge [C], mass [kg]; fields `q`, `m` |
| `Electron()`, `Positron()`, `Proton()` | named particles |
| `Particle(formula, z)` | atom or molecule of charge number `z`; a bare element has its standard atomic weight, `{A}` before it (or `D`, `T`) picks a nuclide; e.g. `Particle(:NO, 1)`, `Particle("{18}O", 3)`, `Particle(:D2, 1)`. `Proton()` is exact ¹H⁺, `Particle(:H, 1)` is not |
| `Population(particle, vdf; n)`, `Population(particle, V; n, kw...)` | population; fields `particle`, `n`, `vdf`; a type `V` builds `vdf = V(particle; kw...)` |
| `Plasma(populations...; B0)` | `Plasma(populations, B0)` with fields `populations`, `B0` |
| `BiMaxwellian(; vth_perp, vth_para, vd_para = 0)` | drifting Maxwellian (`vth`) or bi-Maxwellian |
| `GaussianRing(; vr_perp, w_perp, vth_para, vd_para = 0)` | ring, or ring beam with `vd_para`: `exp(-(v⊥-vr_perp)²/w_perp²) exp(-(v∥-vd_para)²/vth_para²)` |
| `BiKappa(; vth_perp, vth_para, kappa, vd_para = 0)` | coupled bi-kappa, `(1 + v⊥²/A⊥ + (v∥-vd_para)²/A∥)^{-(κ+1)}`, `A = (κ-3/2)vth²` |
| `ProductBiKappa(; vth_perp, vth_para, kappa_perp, kappa_para, loss_cone = 0, vd_para = 0)` | product of independent ⊥ and ∥ kappas; `loss_cone = σ` multiplies f⊥ by `(v⊥²/κ⊥θ⊥²)^σ` |
| `Cold(; vd_para = 0)` | zero-temperature limit |
| `MaxwellJuttner(; theta)` | relativistic Maxwellian `f ∝ exp(-γ/θ)` of temperature `θ = T/(mc²)` |
| `GridVDF(v_perp, v_para, f)` | tabulated `f[i, j]`, bilinear and zero outside the grid; field `n = ∫ f d³v`, e.g. phase-space density → `Population(particle, d; d.n)` |
| `Statistics.mean(vdf)`, `Statistics.cov(vdf)` | mean velocity; velocity covariance `⟨(𝐯 - ⟨𝐯⟩)(𝐯 - ⟨𝐯⟩)ᵀ⟩` |
| `vth_perp_para(vdf)` | gyro-average `(vth_perp, vth_para)` about $𝐁$ |
| `isgyrotropic(vdf)` | whether `vdf` is gyrotropic about $𝐁$ |
| `rand([rng,] vdf[, dims])` | samples `[v₁, v₂, v₃]` |
| `Oriented(vdf, 𝐁; u = 0)` | `vdf` in the frame of vectors `𝐁`, `u` |
| below | public, not exported: `using PlasmaPopulations: gyrofrequency, …` |
| `pdf(vdf, 𝐯)`, `pdf(vdf, v_perp, v_para)` | density at velocity `𝐯` or, if gyrotropic, speeds; normalized to `∫ f d³v = 1` |
| `kappa_widths(d)` | `(θ⊥, θ∥)` of `BiKappa`, `ProductBiKappa` in the forms `v²/(κθ²)` |
| `si_number_density`, `si_magnetic_field`, `si_velocity`, `si_frequency`, `si_charge`, `si_mass`, `si_energy`, `si_phase_space_density` | input adapters to bare SI (energy: eV → J) |
| `C_SI`, `E_SI`, `ME_SI`, `MP_SI`, `U_SI`, `EPS0_SI`, `MU0_SI`, `KB_SI` | CODATA 2018 constants [SI] |
| `gyrofrequency(particle, 𝐁)` | signed $q\|𝐁\|/m$ [rad/s] |
| `plasma_frequency(particle, n)` | $\sqrt{n q²/(ε₀ m)}$ [rad/s] |
| `thermal_speed(particle, T)` | $\sqrt{2T/m}$  [m/s] |
| `inertial_length(particle, n)` | $c/ω_p$ [m] |
| `debye_length(particle, n, T)` | $\sqrt{ε₀T/(n q²)}$ [m] |
| `mass_density(plasma)`, `alfven_speed(plasma)` | total `Σ nm` [kg/m³], `\|B₀\|/√(μ₀ρ)` with `ρ` over charged populations [m/s] |
