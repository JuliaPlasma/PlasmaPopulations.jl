"""
    GridVDF(v_perp, v_para, f)

Gyrotropic distribution tabulated as `f[i, j] = f(v_perp[i], v_para[j])` on ascending grids with `v_perp[1] ≥ 0`,
bilinear between nodes and zero outside them. `f` need not be normalized: field `n = ∫ f d³v` is the number
density [m⁻³] when `f` is a phase-space density [s³ m⁻⁶].
"""
struct GridVDF{T, Vperp <: AbstractVector{T}, Vpara <: AbstractVector{T}, F <: AbstractMatrix, N} <: AbstractVDF
    v_perp::Vperp
    v_para::Vpara
    f::F
    n::N
end

# broadcasting, or `convert` of an integer range, would collect ranges
_si_grid(v) = float(eltype(v) <: Real ? v : si_velocity.(v))

function GridVDF(v_perp, v_para, f)
    vperp, vpara = _si_grid(v_perp), _si_grid(v_para)
    T = promote_type(eltype(vperp), eltype(vpara))
    vperp, vpara = convert(AbstractVector{T}, vperp), convert(AbstractVector{T}, vpara)
    f = si_phase_space_density(f)
    size(f) == (length(vperp), length(vpara)) ||
        throw(DimensionMismatch("f is $(size(f)), grids are $(length(vperp)) × $(length(vpara))"))
    for v in (vperp, vpara)
        length(v) ≥ 2 && all(>(0), diff(v)) || throw(ArgumentError("GridVDF needs strictly ascending grids of ≥ 2 nodes"))
    end
    minimum(vperp) ≥ 0 || throw(DomainError(minimum(vperp), "GridVDF needs v_perp ≥ 0"))
    n = _integral(vperp, vpara, f, one, one)
    iszero(n) && throw(ArgumentError("GridVDF needs ∫ f d³v ≠ 0"))
    return GridVDF(vperp, vpara, f, n)
end

# ∫ gperp(v⊥) gpara(v∥) f d³v over the bilinear interpolant, so that moments and `pdf` describe the same distribution.
function _integral(vperp, vpara, f, gperp, gpara)
    wperp = _like(f, _hat_weights(v -> v * gperp(v), vperp))
    wpara = _like(f, _hat_weights(gpara, vpara))
    return 2 * (π * (wperp' * f * wpara))
end

# weights on `f`'s device, as a range grid with device `f` (or vice versa) cannot multiply
_like(f, w) = copyto!(similar(f, eltype(w), length(w)), w)
_moment(d::GridVDF{T}, gperp, gpara) where {T} = T(_integral(d.v_perp, d.v_para, d.f, gperp, gpara) / d.n)

# w[i] = ∫ φᵢ g dx for the piecewise-linear hats φᵢ on x, by 3-point Gauss–Legendre per interval: exact for g of degree ≤ 4.
# Whole-array operations only, so that device grids work.
function _hat_weights(g, x::AbstractVector{T}) where {T}
    lr = _interval_weights.(g, @view(x[1:(end - 1)]), @view(x[2:end]))
    z = fill!(similar(x, T, 1), 0)
    return vcat(first.(lr), z) .+ vcat(z, last.(lr))
end

# (∫ φ_left g, ∫ φ_right g) over [a, b]
function _interval_weights(g, a::T, b::T) where {T}
    s = sqrt(T(3) / 5)
    l = r = zero(T)
    for (t, wt) in ((-s, T(5) / 9), (zero(T), T(8) / 9), (s, T(5) / 9))
        ξ = (a + b) / 2 + t * (b - a) / 2
        gw = wt / 2 * g(ξ)
        l += gw * (b - ξ)
        r += gw * (ξ - a)
    end
    return l, r
end

function _pdf(d::GridVDF, _, v_perp, v_para)
    x, y, f = d.v_perp, d.v_para, d.f
    a, b = abs(v_perp), v_para
    R = typeof(zero(eltype(f)) * zero(a) * zero(b) / d.n)
    (x[1] ≤ a ≤ x[end] && y[1] ≤ b ≤ y[end]) || return zero(R)
    i, j = min(searchsortedlast(x, a), length(x) - 1), min(searchsortedlast(y, b), length(y) - 1)
    s, t = (a - x[i]) / (x[i + 1] - x[i]), (b - y[j]) / (y[j + 1] - y[j])
    fv = (1 - s) * ((1 - t) * f[i, j] + t * f[i, j + 1]) + s * ((1 - t) * f[i + 1, j] + t * f[i + 1, j + 1])
    return R(fv / d.n)
end

isgyrotropic(::Type{<:GridVDF}) = true
Statistics.mean(d::GridVDF) = _mean_para(_moment(d, one, identity))
function vth_perp_para(d::GridVDF)
    vd = _moment(d, one, identity)
    return (sqrt(_moment(d, abs2, one)), sqrt(2 * _moment(d, one, u -> (u - vd)^2)))
end

Base.show(io::IO, d::GridVDF) =
    print(io, "GridVDF(", length(d.v_perp), " × ", length(d.v_para), " grid, n = ", d.n, ")")
