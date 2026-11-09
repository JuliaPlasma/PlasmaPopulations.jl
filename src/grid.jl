"""
    GridVDF(v_perp, v_para, f)

Gyrotropic distribution tabulated as `f[i, j] = f(v_perp[i], v_para[j])` on ascending grids with `v_perp[1] ≥ 0`,
bilinear between nodes and zero outside them. `f` need not be normalized: field `n = ∫ f d³v` is the number
density [m⁻³] when `f` is a phase-space density [s³ m⁻⁶].
"""
struct GridVDF{T, F <: AbstractMatrix, N} <: AbstractVDF
    v_perp::Vector{T}
    v_para::Vector{T}
    f::F
    n::N
end

function GridVDF(v_perp, v_para, f)
    vperp, vpara = map(si_velocity, v_perp), map(si_velocity, v_para)
    T = float(promote_type(eltype(vperp), eltype(vpara)))
    vperp, vpara, f = convert(Vector{T}, vperp), convert(Vector{T}, vpara), si_phase_space_density(f)
    size(f) == (length(vperp), length(vpara)) ||
        throw(DimensionMismatch("f is $(size(f)), grids are $(length(vperp)) × $(length(vpara))"))
    for v in (vperp, vpara)
        length(v) ≥ 2 && all(>(0), diff(v)) || throw(ArgumentError("GridVDF needs strictly ascending grids of ≥ 2 nodes"))
    end
    vperp[1] ≥ 0 || throw(DomainError(vperp[1], "GridVDF needs v_perp ≥ 0"))
    n = _integral(vperp, vpara, f, one, one)
    iszero(n) && throw(ArgumentError("GridVDF needs ∫ f d³v ≠ 0"))
    return GridVDF(vperp, vpara, f, n)
end

# ∫ gperp(v⊥) gpara(v∥) f d³v over the bilinear interpolant, so that moments and `pdf` describe the same distribution.
_integral(vperp, vpara, f, gperp, gpara) = 2π * (_hat_weights(v -> v * gperp(v), vperp)' * f * _hat_weights(gpara, vpara))
_moment(d::GridVDF{T}, gperp, gpara) where {T} = T(_integral(d.v_perp, d.v_para, d.f, gperp, gpara) / d.n)

# w[i] = ∫ φᵢ g dx for the piecewise-linear hats φᵢ on x, by 3-point Gauss–Legendre per interval: exact for g of degree ≤ 4.
function _hat_weights(g, x::AbstractVector{T}) where {T}
    w = zeros(T, length(x))
    s = sqrt(T(3) / 5)
    for i in 1:(length(x) - 1)
        a, b = x[i], x[i + 1]
        h = b - a
        for (t, wt) in ((-s, T(5) / 9), (zero(T), T(8) / 9), (s, T(5) / 9))
            ξ = (a + b) / 2 + t * h / 2
            gw = wt / 2 * g(ξ)
            w[i] += gw * (b - ξ)
            w[i + 1] += gw * (ξ - a)
        end
    end
    return w
end

function _pdf(d::GridVDF, v_perp, v_para)
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
