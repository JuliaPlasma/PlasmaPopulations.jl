"Velocity distribution, normalized to `∫ f d³v = 1`. "
abstract type AbstractVDF end

Base.broadcastable(d::AbstractVDF) = Ref(d)

function pdf end

isgyrotropic(d) = isgyrotropic(typeof(d))
isgyrotropic(::Type) = false

# Gyrotropic models implement `pdf(d, v_perp, v_para)`, even in `v_perp`, and `vth_perp_para`; others `pdf(d, 𝐯)` and `cov`.

pdf(d::AbstractVDF, v) = isgyrotropic(d) ? pdf(d, sqrt(v[1]^2 + v[2]^2), v[3]) : throw(MethodError(pdf, (d, v)))
pdf(d, v_perp, v_para) = _pdf(d, si_velocity(v_perp), si_velocity(v_para))

Statistics.cov(d::AbstractVDF) = isgyrotropic(d) ? _gyrotropic_cov(vth_perp_para(d)...) : throw(MethodError(cov, (d,)))
vth_perp_para(d::AbstractVDF) = isgyrotropic(d) ? throw(MethodError(vth_perp_para, (d,))) : _vth_perp_para(cov(d))

function _gyrotropic_cov(vth_perp, vth_para)
    a, b = vth_perp^2 / 2, vth_para^2 / 2
    z = zero(a)
    return SMatrix{3, 3}(a, z, z, z, a, z, z, z, b)
end
_vth_perp_para(C) = (sqrt(C[1, 1] + C[2, 2]), sqrt(2 * C[3, 3]))

# A plain `sqrt(π)`, `π^p` or integer `κ` would widen the speeds' float type, or cap its precision.
_speed_float(d, x) = x * one(float(d.vth_para))

_isotropic_vth(vth) = vth
_isotropic_vth(::Nothing) = throw(ArgumentError("give `vth`, or both `vth_perp` and `vth_para`"))

_si_speeds(speeds...) = promote(map(si_velocity, speeds)...)

_maxwellian_para(d, v_para) = exp(-((v_para - d.vd_para) / d.vth_para)^2) / sqrt(π * d.vth_para^2)
_rand_para(rng, d, T) = d.vd_para + d.vth_para / sqrt(T(2)) * randn(rng, T)

function _with_gyrophase(rng, v_perp, v_para, T)
    sinφ, cosφ = sincospi(2 * rand(rng, T))
    return SVector(v_perp * cosφ, v_perp * sinφ, v_para)
end

Random.rand(rng::AbstractRNG, d::AbstractVDF, dims::Dims) = [rand(rng, d) for _ in CartesianIndices(dims)]

"""
    BiMaxwellian(; vth, vd_para = 0)
    BiMaxwellian(; vth_perp, vth_para, vd_para = 0)

Drifting bi-Maxwellian.

    f ∝ exp(-v⊥²/vth⊥²) · exp(-(v∥ - vd_para)²/vth∥²)
"""
struct BiMaxwellian{T} <: AbstractVDF
    vth_perp::T
    vth_para::T
    vd_para::T
    function BiMaxwellian(vth_perp, vth_para, vd_para)
        speeds = _si_speeds(vth_perp, vth_para, vd_para)
        return new{eltype(speeds)}(speeds...)
    end
end

BiMaxwellian(; vth = nothing, vth_perp = _isotropic_vth(vth), vth_para = _isotropic_vth(vth), vd_para = zero(vth_para)) =
    BiMaxwellian(vth_perp, vth_para, vd_para)

function _pdf(d::BiMaxwellian, v_perp, v_para)
    a = d.vth_perp
    return exp(-(v_perp / a)^2) / (π * a^2) * _maxwellian_para(d, v_para)
end

function Random.rand(rng::AbstractRNG, d::BiMaxwellian)
    T = float(typeof(d.vth_perp))
    sperp = d.vth_perp / sqrt(T(2))
    return SVector(sperp * randn(rng, T), sperp * randn(rng, T), _rand_para(rng, d, T))
end

"""
    GaussianRing(; vr_perp, w_perp, vth_para, vd_para = 0)

Ring of speed `vr_perp ≥ 0` with a Gaussian radial profile of 1/e half-width `w_perp > 0`; a ring beam when `vd_para ≠ 0`.

    f ∝ exp(-(v⊥ - vr_perp)²/w_perp²) · exp(-(v∥ - vd_para)²/vth∥²)

`w_perp` is a shape parameter.
"""
struct GaussianRing{T} <: AbstractVDF
    vr_perp::T
    w_perp::T
    vth_para::T
    vd_para::T
    function GaussianRing(vr_perp, w_perp, vth_para, vd_para)
        speeds = _si_speeds(vr_perp, w_perp, vth_para, vd_para)
        speeds[1] ≥ 0 || throw(DomainError(speeds[1], "GaussianRing needs vr_perp ≥ 0"))
        speeds[2] > 0 || throw(DomainError(speeds[2], "GaussianRing needs w_perp > 0"))
        return new{eltype(speeds)}(speeds...)
    end
end

GaussianRing(; vr_perp, w_perp, vth_para, vd_para = zero(vth_para)) = GaussianRing(vr_perp, w_perp, vth_para, vd_para)

# With g = exp(-(v - vr_perp)²/w²) and r = vr_perp/w: ∫ v g dv = w²(e + rE)/2,  ∫ v³ g dv = w⁴(2e(1 + r²) + (3r + 2r³)E)/4
function vth_perp_para(d::GaussianRing)
    r = d.vr_perp / d.w_perp
    e, E = exp(-r^2), sqrt(π * one(r)) * erfc(-r)
    return (d.w_perp * sqrt((2e * (1 + r^2) + (3r + 2r^3) * E) / (2 * (e + r * E))), d.vth_para)
end

function _pdf(d::GaussianRing, v_perp, v_para)
    a, r = d.w_perp, d.vr_perp / d.w_perp
    fperp = exp(-((abs(v_perp) - d.vr_perp) / a)^2) / (π * a^2 * (exp(-r^2) + sqrt(π * one(r)) * r * erfc(-r)))
    return fperp * _maxwellian_para(d, v_para)
end

# x ∝ x exp(-(x - r)²) on x > 0, r ≥ 0, by rejection from (r + |x - r|) exp(-(x - r)²): a mixture of
# a Gaussian (weight r√π) and a two-sided Rayleigh (weight 1). Acceptance is at least 1/2.
function _rand_ring(rng::AbstractRNG, r::T) where {T}
    w = sqrt(π * r^2)
    while true
        y = rand(rng, T) * (w + 1) < w ? randn(rng, T) / sqrt(T(2)) :
            (rand(rng, Bool) ? 1 : -1) * sqrt(randexp(rng, T))
        x = r + y
        x > 0 && rand(rng, T) * (r + abs(y)) < x && return x
    end
end

function Random.rand(rng::AbstractRNG, d::GaussianRing)
    T = float(typeof(d.w_perp))
    v_perp = d.w_perp * _rand_ring(rng, T(d.vr_perp / d.w_perp))
    return _with_gyrophase(rng, v_perp, _rand_para(rng, d, T), T)
end

"""
    BiKappa(; vth, kappa, vd_para = 0)
    BiKappa(; vth_perp, vth_para, kappa, vd_para = 0)

Coupled bi-kappa, `κ > 3/2`, with κ-independent temperatures.

    f ∝ (1 + v⊥²/A⊥ + (v∥ - vd_para)²/A∥)^{-(κ+1)},   A = (κ - 3/2)·vth²
"""
struct BiKappa{T, K} <: AbstractVDF
    vth_perp::T
    vth_para::T
    kappa::K
    vd_para::T
    function BiKappa(vth_perp, vth_para, kappa, vd_para)
        kappa > 3 / 2 || throw(DomainError(kappa, "BiKappa needs κ > 3/2"))
        v_perp, v_para, vd = _si_speeds(vth_perp, vth_para, vd_para)
        return new{typeof(v_perp), typeof(kappa)}(v_perp, v_para, kappa, vd)
    end
end

BiKappa(; vth = nothing, vth_perp = _isotropic_vth(vth), vth_para = _isotropic_vth(vth), kappa, vd_para = zero(vth_para)) =
    BiKappa(vth_perp, vth_para, kappa, vd_para)

function _pdf(d::BiKappa, v_perp, v_para)
    κ = _speed_float(d, d.kappa)
    Aperp, Apara = (κ - 3 // 2) .* (d.vth_perp^2, d.vth_para^2)
    C = exp(loggamma(κ + 1) - loggamma(κ - 1 // 2)) / (π * Aperp * sqrt(π * Apara))
    return C * (1 + v_perp^2 / Aperp + (v_para - d.vd_para)^2 / Apara)^(-(κ + 1))
end

# Multivariate Student-t: x = √(A/ξ)·Z with ξ ~ χ²_{2κ-1} = 2·Gamma(κ - 1/2)
function Random.rand(rng::AbstractRNG, d::BiKappa)
    κ = _speed_float(d, d.kappa)
    T = typeof(κ)
    s = sqrt((κ - 3 // 2) / (2 * _rand_gamma(rng, κ - 1 // 2)))
    sperp, spara = s .* (d.vth_perp, d.vth_para)
    return SVector(sperp * randn(rng, T), sperp * randn(rng, T), d.vd_para + spara * randn(rng, T))
end

"""
    ProductBiKappa(; vth, kappa_perp, kappa_para, loss_cone = 0, vd_para = 0)
    ProductBiKappa(; vth_perp, vth_para, kappa_perp, kappa_para, loss_cone = 0, vd_para = 0)

Separable product bi-kappa with independent indices `κ⊥ > 1`, `κ∥ > 1/2` and a kappa
loss cone of index `σ = loss_cone ≥ 0` (Summers–Thorne; `σ = 0` is no loss cone), with κ-independent temperatures.

    f ∝ x^σ (1 + x)^{-(κ⊥+σ+1)} · (1 + (v∥ - vd_para)²/(κ∥θ∥²))^{-(κ∥+1)},   x = v⊥²/(κ⊥θ⊥²)
    θ⊥² = (1 - 1/κ⊥)·vth⊥²/(1 + σ),   θ∥² = (1 - 1/(2κ∥))·vth∥²
"""
struct ProductBiKappa{T, K} <: AbstractVDF
    vth_perp::T
    vth_para::T
    kappa_perp::K
    kappa_para::K
    loss_cone::K
    vd_para::T
    function ProductBiKappa(vth_perp, vth_para, kappa_perp, kappa_para, loss_cone, vd_para)
        kappa_perp > 1 || throw(DomainError(kappa_perp, "ProductBiKappa needs κperp > 1"))
        kappa_para > 1 / 2 || throw(DomainError(kappa_para, "ProductBiKappa needs κpara > 1/2"))
        loss_cone ≥ 0 || throw(DomainError(loss_cone, "ProductBiKappa needs loss_cone ≥ 0"))
        v_perp, v_para, vd = _si_speeds(vth_perp, vth_para, vd_para)
        κperp, κpara, σ = promote(kappa_perp, kappa_para, loss_cone)
        return new{typeof(v_perp), typeof(κperp)}(v_perp, v_para, κperp, κpara, σ, vd)
    end
end

ProductBiKappa(; vth = nothing, vth_perp = _isotropic_vth(vth), vth_para = _isotropic_vth(vth), kappa_perp, kappa_para, loss_cone = 0, vd_para = zero(vth_para)) =
    ProductBiKappa(vth_perp, vth_para, kappa_perp, kappa_para, loss_cone, vd_para)

function kappa_widths(d::BiKappa)
    s = sqrt(1 - 3 / (2 * _speed_float(d, d.kappa)))
    return (s * d.vth_perp, s * d.vth_para)
end

function kappa_widths(d::ProductBiKappa)
    κperp, κpara, σ = _speed_float.(Ref(d), (d.kappa_perp, d.kappa_para, d.loss_cone))
    return (sqrt((1 - 1 / κperp) / (1 + σ)) * d.vth_perp, sqrt(1 - 1 / (2κpara)) * d.vth_para)
end

function _pdf(d::ProductBiKappa, v_perp, v_para)
    κperp, κpara, σ = _speed_float.(Ref(d), (d.kappa_perp, d.kappa_para, d.loss_cone))
    θperp, θpara = kappa_widths(d)
    x = v_perp^2 / (κperp * θperp^2)
    # 1 / ∫2πv⊥ x^σ (1+x)^{-(κ+σ+1)} dv⊥ = Γ(κ+σ+1) / (πκθ² Γ(σ+1) Γ(κ))
    Cperp = exp(loggamma(κperp + σ + 1) - loggamma(σ + 1) - loggamma(κperp)) / (π * κperp * θperp^2)
    fperp = Cperp * x^σ * (1 + x)^(-(κperp + σ + 1))
    Cpara = exp(loggamma(κpara + 1) - loggamma(κpara + 1 // 2)) / (sqrt(π * κpara) * θpara)
    fpara = Cpara * (1 + (v_para - d.vd_para)^2 / (κpara * θpara^2))^(-(κpara + 1))
    return fperp * fpara
end

# ⊥: x = v⊥²/(κθ²) ~ BetaPrime(σ+1, κ) = Gamma(σ+1)/Gamma(κ); ∥: Student-t with 2κ∥+1 dof
function Random.rand(rng::AbstractRNG, d::ProductBiKappa)
    κperp, κpara, σ = _speed_float.(Ref(d), (d.kappa_perp, d.kappa_para, d.loss_cone))
    T = typeof(κperp)
    θperp, θpara = kappa_widths(d)
    v_perp = θperp * sqrt(κperp * _rand_gamma(rng, σ + 1) / _rand_gamma(rng, κperp))
    spara = θpara * sqrt(κpara / (2 * _rand_gamma(rng, κpara + 1 // 2)))
    return _with_gyrophase(rng, v_perp, d.vd_para + spara * randn(rng, T), T)
end

"""
    DoryGuestHarris(; vth, loss_cone, vd_para = 0)
    DoryGuestHarris(; vth_perp, vth_para, loss_cone, vd_para = 0)

Maxwellian loss cone of index `j = loss_cone ≥ 0` (Dory, Guest & Harris 1965, PRL 14, 131); `j = 0` is `BiMaxwellian`.

    f ∝ (v⊥/θ⊥)^{2j} exp(-v⊥²/θ⊥²) · exp(-(v∥ - vd_para)²/vth∥²),   θ⊥² = vth⊥²/(1 + j)
"""
struct DoryGuestHarris{T, K} <: AbstractVDF
    vth_perp::T
    vth_para::T
    loss_cone::K
    vd_para::T
    function DoryGuestHarris(vth_perp, vth_para, loss_cone, vd_para)
        loss_cone ≥ 0 || throw(DomainError(loss_cone, "DoryGuestHarris needs loss_cone ≥ 0"))
        v_perp, v_para, vd = _si_speeds(vth_perp, vth_para, vd_para)
        return new{typeof(v_perp), typeof(loss_cone)}(v_perp, v_para, loss_cone, vd)
    end
end

DoryGuestHarris(; vth = nothing, vth_perp = _isotropic_vth(vth), vth_para = _isotropic_vth(vth), loss_cone, vd_para = zero(vth_para)) =
    DoryGuestHarris(vth_perp, vth_para, loss_cone, vd_para)

function _pdf(d::DoryGuestHarris, v_perp, v_para)
    j = _speed_float(d, d.loss_cone)
    θ² = d.vth_perp^2 / (1 + j)
    x = v_perp^2 / θ²
    return x^j * exp(-x - loggamma(j + 1)) / (π * θ²) * _maxwellian_para(d, v_para)
end

function Random.rand(rng::AbstractRNG, d::DoryGuestHarris)
    j = _speed_float(d, d.loss_cone)
    T = typeof(j)
    v_perp = d.vth_perp * sqrt(_rand_gamma(rng, j + 1) / (1 + j))
    return _with_gyrophase(rng, v_perp, _rand_para(rng, d, T), T)
end

"""
    SubtractedMaxwellian(; vth, delta, beta, vd_para = 0)
    SubtractedMaxwellian(; vth_perp, vth_para, delta, beta, vd_para = 0)

Maxwellian loss cone of fill `0 ≤ Δ = delta ≤ 1` and width `0 < β = beta < 1` (Ashour-Abdalla & Kennel 1978, JGR 83, 1531); `Δ = 1` is `BiMaxwellian`.

    f ∝ [Δ e^{-x} + (1 - Δ)(e^{-x} - e^{-x/β})/(1 - β)] · exp(-(v∥ - vd_para)²/vth∥²),   x = v⊥²/a⊥²,   a⊥² = vth⊥²/(1 + β(1 - Δ))
"""
struct SubtractedMaxwellian{T, K} <: AbstractVDF
    vth_perp::T
    vth_para::T
    delta::K
    beta::K
    vd_para::T
    function SubtractedMaxwellian(vth_perp, vth_para, delta, beta, vd_para)
        0 ≤ delta ≤ 1 || throw(DomainError(delta, "SubtractedMaxwellian needs 0 ≤ delta ≤ 1"))
        0 < beta < 1 || throw(DomainError(beta, "SubtractedMaxwellian needs 0 < beta < 1"))
        v_perp, v_para, vd = _si_speeds(vth_perp, vth_para, vd_para)
        Δ, β = promote(delta, beta)
        return new{typeof(v_perp), typeof(Δ)}(v_perp, v_para, Δ, β, vd)
    end
end

SubtractedMaxwellian(; vth = nothing, vth_perp = _isotropic_vth(vth), vth_para = _isotropic_vth(vth), delta, beta, vd_para = zero(vth_para)) =
    SubtractedMaxwellian(vth_perp, vth_para, delta, beta, vd_para)

function _pdf(d::SubtractedMaxwellian, v_perp, v_para)
    Δ, β = _speed_float.(Ref(d), (d.delta, d.beta))
    a² = d.vth_perp^2 / (1 + β * (1 - Δ))
    x = v_perp^2 / a²
    fperp = (Δ * exp(-x) + (1 - Δ) * (exp(-x) - exp(-x / β)) / (1 - β)) / (π * a²)
    return fperp * _maxwellian_para(d, v_para)
end

# The loss-cone part of x = v⊥²/a⊥², (e^{-x} - e^{-x/β})/(1 - β), is the density of Exp(1) + β·Exp(1).
function Random.rand(rng::AbstractRNG, d::SubtractedMaxwellian)
    Δ, β = _speed_float.(Ref(d), (d.delta, d.beta))
    T = typeof(β)
    x = randexp(rng, T)
    rand(rng, T) < Δ || (x += β * randexp(rng, T))
    v_perp = d.vth_perp * sqrt(x / (1 + β * (1 - Δ)))
    return _with_gyrophase(rng, v_perp, _rand_para(rng, d, T), T)
end

const _MomentVth = Union{BiMaxwellian, BiKappa, ProductBiKappa, DoryGuestHarris, SubtractedMaxwellian}

function (::Type{V})(p::Particle; T = nothing, T_perp = T, T_para = T, kw...) where {V <: Union{_MomentVth, GaussianRing}}
    temperatures = pairs((; vth_perp = T_perp, vth_para = T_para))
    return V(; (k => thermal_speed(p, t) for (k, t) in temperatures if !isnothing(t))..., kw...)
end

"""
    Cold(; vd_para = 0)

Zero-temperature limit, drifting at `vd_para` along the field.
"""
struct Cold{T} <: AbstractVDF
    vd_para::T
    Cold(vd_para) = (v = si_velocity(vd_para); new{typeof(v)}(v))
end
Cold(; vd_para = 0.0) = Cold(vd_para)
Cold(::Particle; kw...) = Cold(; kw...)

vth_perp_para(d::Cold) = (zero(d.vd_para), zero(d.vd_para))
Random.rand(::AbstractRNG, d::Cold) = mean(d)

struct MaxwellJuttner{T} <: AbstractVDF
    theta::T
    function MaxwellJuttner(theta)
        theta > 0 || throw(DomainError(theta, "MaxwellJuttner needs theta > 0"))
        return new{typeof(theta)}(theta)
    end
end
MaxwellJuttner(; theta) = MaxwellJuttner(theta)
MaxwellJuttner(p::Particle; T) = MaxwellJuttner(si_energy(T) / (p.m * C_SI^2))

# K₂, K₃ scaled by exp(1/θ), which cancels between the normalization and exp(-γ/θ) = exp(-(γ - 1)/θ) exp(-1/θ).
function _pdf(d::MaxwellJuttner, v_perp, v_para)
    θ = float(d.theta)
    c = convert(typeof(θ), C_SI)
    x = (v_perp^2 + v_para^2) / c^2
    return exp(-x / (1 + sqrt(1 + x)) / θ) / (4 * oftype(θ, π) * c^3 * θ * _besselkx(2, 1 / θ))
end

# ⟨u_i²⟩ = c²θ K₃(1/θ)/K₂(1/θ)
function vth_perp_para(d::MaxwellJuttner)
    θ = float(d.theta)
    v = convert(typeof(θ), C_SI) * sqrt(2θ * _besselkx(3, 1 / θ) / _besselkx(2, 1 / θ))
    return (v, v)
end

Statistics.mean(d::MaxwellJuttner) = zero(SVector{3, typeof(float(d.theta))})

# T/m per (⊥, ∥) degree of freedom, from the pressure P = n m ⟨δ𝐮 δ𝐯⟩: `cov` is its nonrelativistic limit.
_temperature_per_mass(d) = vth_perp_para(d) .^ 2 ./ 2
_temperature_per_mass(d::MaxwellJuttner) = (c2θ = convert(typeof(float(d.theta)), C_SI)^2 * d.theta; (c2θ, c2θ))

const _FieldAligned = Union{_MomentVth, GaussianRing, Cold}
isgyrotropic(::Type{<:Union{_FieldAligned, MaxwellJuttner}}) = true
Statistics.mean(d::_FieldAligned) = _mean_para(d.vd_para)
vth_perp_para(d::_MomentVth) = (d.vth_perp, d.vth_para)
_mean_para(vd) = SVector(zero(vd), zero(vd), vd)

# Marsaglia & Tsang (2000) Gamma(α, 1) sampler, α ≥ 1
function _rand_gamma(rng::AbstractRNG, α::T) where {T}
    α ≥ 1 || throw(DomainError(α, "Gamma sampler needs α ≥ 1"))
    d = α - 1 // 3
    c = 1 / sqrt(9d)
    while true
        Z = randn(rng, T)
        Z > -1 / c || continue
        v = (1 + c * Z)^3
        log(rand(rng, T)) < Z^2 / 2 + d * (1 - v + log(v)) && return d * v
    end
end
