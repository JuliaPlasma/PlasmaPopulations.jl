"""
    GyrotropicGridVDF(v_perp, v_para, f)

Grids strictly ascending, ≥ 2 nodes, `v_perp[1] ≥ 0`.
"""
struct GyrotropicGridVDF{T, Vperp <: AbstractVector{T}, Vpara <: AbstractVector{T}, F <: AbstractMatrix, N} <: AbstractVDF
    v_perp::Vperp
    v_para::Vpara
    f::F
    n::N
end

Base.@deprecate_binding GridVDF GyrotropicGridVDF

"""
    CartesianGridVDF(vx, vy, vz, f)

Axes strictly ascending, ≥ 2 nodes.
"""
struct CartesianGridVDF{T, Vx <: AbstractVector{T}, Vy <: AbstractVector{T}, Vz <: AbstractVector{T}, F <: AbstractArray{<:Any, 3}, N} <: AbstractVDF
    vx::Vx
    vy::Vy
    vz::Vz
    f::F
    n::N
end

const _GridVDF = Union{GyrotropicGridVDF, CartesianGridVDF}

_grids(d::GyrotropicGridVDF) = (d.v_perp, d.v_para)
_grids(d::CartesianGridVDF) = (d.vx, d.vy, d.vz)

# d³v as a product of per-axis factors
_measure(::Type{<:GyrotropicGridVDF}) = (_perp_measure, one)
_perp_measure(v) = 2 * (π * v)
_measure(::Type{<:CartesianGridVDF}) = (one, one, one)

# broadcasting, or `convert` of an integer range, would collect ranges
_si_grid(v) = float(eltype(v) <: Real ? v : si_velocity.(v))

function _si_grids(f, grids...)
    grids = map(_si_grid, grids)
    T = promote_type(map(eltype, grids)...)
    grids = map(x -> convert(AbstractVector{T}, x), grids)
    size(f) == map(length, grids) ||
        throw(DimensionMismatch("f is $(size(f)), grids are $(join(map(length, grids), " × "))"))
    for x in grids
        h = convert(Array, x)  # grids are short: on the device, kernel compile and launch would dwarf the work
        length(h) ≥ 2 && all(i -> h[i] < h[i + 1], 1:(length(h) - 1)) || throw(ArgumentError("grid VDFs need strictly ascending grids of ≥ 2 nodes"))
    end
    return grids
end

# ∫ f d³v, the density that `pdf` normalizes by
function _density(V, grids, f)
    n = only(_integrals(grids, f, map(tuple, _measure(V))))
    isfinite(n) && n > 0 || throw(ArgumentError("grid VDFs need a finite, positive ∫ f d³v"))
    return n
end

function GyrotropicGridVDF(v_perp, v_para, f)
    f = si_phase_space_density(f)
    grids = _si_grids(f, v_perp, v_para)
    v0 = first(convert(Array, grids[1]))  # the minimum, as the grid ascends
    v0 ≥ 0 || throw(DomainError(v0, "GyrotropicGridVDF needs v_perp ≥ 0"))
    return GyrotropicGridVDF(grids..., f, _density(GyrotropicGridVDF, grids, f))
end

function CartesianGridVDF(vx, vy, vz, f)
    f = si_phase_space_density(f)
    grids = _si_grids(f, vx, vy, vz)
    return CartesianGridVDF(grids..., f, _density(CartesianGridVDF, grids, f))
end

# M[a, b, …] = ∫ fns[1][a](x₁) fns[2][b](x₂) ⋯ f dx₁ dx₂ ⋯ for all products at once, over the multilinear interpolant so that
# moments and `pdf` describe the same distribution.
function _integrals(grids, f, fns; origins = map(zero ∘ eltype, grids))
    ws = map((g, x, o) -> map(gₖ -> _like(f, _hat_weights(gₖ, x, o)), g), fns, grids, origins)
    return Array(_contract(f, ws...))
end

# contract `f`'s first axis with each weight vector in `w`, appending them as the last axis. Matrix–vector products, so that
# device arrays work; one per vector, as BLAS runs a thin matrix–matrix product slower than as many matrix–vector ones.
_contract(f) = f
function _contract(f, w, ws...)
    F = transpose(reshape(f, size(f, 1), :))
    return _contract(reshape(hcat(map(wₖ -> F * wₖ, w)...), Base.tail(size(f))..., length(w)), ws...)
end

# weights on `f`'s device, as a range grid with device `f` (or vice versa) cannot multiply
_like(f, w) = copyto!(similar(f, eltype(w), length(w)), w)

# M[a, b, …] = ⟨fns[1][a](x₁) fns[2][b](x₂) ⋯⟩, each axis's `fns` taking coordinates relative to its origin and the measure
# absolute ones
function _moments(d::_GridVDF, fns...; origins = map(zero ∘ eltype, _grids(d)))
    grids = _grids(d)
    M = _integrals(grids, d.f, map((m, g, o) -> map(gₖ -> ξ -> m(ξ + o) * gₖ(ξ), g), _measure(typeof(d)), fns, origins); origins)
    return eltype(first(grids)).(M ./ d.n)
end

# w[i] = ∫ φᵢ(x) g(x - origin) dx for the linear hats, by 3-point Gauss–Legendre per interval: exact for g of degree ≤ 4.
# Whole-array operations only, so that device grids work.
function _hat_weights(g, x::AbstractVector{T}, origin) where {T}
    lr = _interval_weights.(g, @view(x[1:(end - 1)]), @view(x[2:end]), origin)
    z = fill!(similar(x, T, 1), 0)
    return vcat(first.(lr), z) .+ vcat(z, last.(lr))
end

# Use the original interval width: subtracting the origin from both endpoints can collapse distant narrow cells.
function _interval_weights(g, a::T, b::T, origin = zero(T)) where {T}
    s = sqrt(T(3) / 5)
    l = r = zero(T)
    for (t, wt) in ((-s, T(5) / 9), (zero(T), T(8) / 9), (s, T(5) / 9))
        u = (1 + t) / 2
        gw = wt / 2 * g(muladd(u, b - a, a - origin)) * (b - a)
        l += gw * (1 - u)
        r += gw * u
    end
    return l, r
end

# node index `i` and fraction `s` of `a` in the cell `[x[i], x[i + 1]]`; `a` inside the grid
@inline function _cell(x, a)
    i = min(searchsortedlast(x, a), length(x) - 1)
    return i, (a - x[i]) / (x[i + 1] - x[i])
end

_inside(x, a) = x[1] ≤ a ≤ x[end]

# normalized multilinear interpolant at point `p`, zero outside the grids
function _interp(d::_GridVDF, p)
    R = typeof(zero(eltype(d.f)) * prod(zero, p) / d.n)
    grids = _grids(d)
    all(map(_inside, grids, p)) || return zero(R)
    return R(_lerp(d.f, firstindex(d.f), map(_cell, grids, p), Base.front(cumprod((1, size(d.f)...)))) / d.n)
end

# recursion over axes on the linear index `o`, as a growing index tuple stops specializing and allocates beyond 2 axes
@inline _lerp(f, o, ::Tuple{}, ::Tuple{}) = f[o]
@inline function _lerp(f, o, cells, strides)
    (i, s), st = first(cells), first(strides)
    o += (i - 1) * st
    return (1 - s) * _lerp(f, o, Base.tail(cells), Base.tail(strides)) + s * _lerp(f, o + st, Base.tail(cells), Base.tail(strides))
end

_pdf(d::GyrotropicGridVDF, _, v_perp, v_para) = _interp(d, (abs(v_perp), v_para))
pdf(d::CartesianGridVDF, v) = _interp(d, (si_velocity(v[1]), si_velocity(v[2]), si_velocity(v[3])))

isgyrotropic(::Type{<:GyrotropicGridVDF}) = true
Statistics.mean(d::GyrotropicGridVDF) = _along_z(only(_moments(d, (one,), (identity,))))
function vth_perp_para(d::GyrotropicGridVDF)
    vd = only(_moments(d, (one,), (identity,)))
    M = _moments(d, (one, abs2), (one, identity, abs2); origins = (zero(vd), vd))
    return (sqrt(M[2, 1]), sqrt(2 * (M[1, 3] - M[1, 2]^2)))
end

# M[1 + a, 1 + b, 1 + c] = ⟨vx^a vy^b vz^c⟩
function Statistics.mean(d::CartesianGridVDF)
    M = _moments(d, ntuple(_ -> (one, identity), 3)...)
    return SVector(M[2, 1, 1], M[1, 2, 1], M[1, 1, 2])
end

# Shift the axes before placing quadrature nodes: subcell offsets can be smaller than one ulp of the drift.
function Statistics.cov(d::CartesianGridVDF)
    μ = mean(d)
    M = _moments(d, ntuple(_ -> (one, identity, abs2), 3)...; origins = Tuple(μ))
    δ = SVector(M[2, 1, 1], M[1, 2, 1], M[1, 1, 2])
    return SMatrix{3, 3}(
        M[3, 1, 1], M[2, 2, 1], M[2, 1, 2],
        M[2, 2, 1], M[1, 3, 1], M[1, 2, 2],
        M[2, 1, 2], M[1, 2, 2], M[1, 1, 3],
    ) - δ * δ'
end

# Draws from the multilinear interpolant f(𝐱) ∏ₖ mₖ(xₖ) = Σₙ f[n] ∏ₖ φₙₖ(xₖ) mₖ(xₖ), with φₙₖ the hat of node nₖ and mₖ the
# measure: a mixture over nodes n of weight f[n] ∏ₖ ∫ φₙₖ mₖ, each a product of 1-D densities φₙₖ mₖ, of which a draw picks
# the left or right half by its integral. Tables on the host, as draws are scalar.
struct _GridSampler{D, G, W, C}
    d::D
    grids::G
    w::W  # integrals of the two hats on each axis interval
    cdf::C
end

# (∫ φᵢ m) over the intervals left and right of node i
_halves(w, i) = (i > 1 ? w[i - 1][2] : zero(w[1][2]), i ≤ length(w) ? w[i][1] : zero(w[1][1]))

function _sampler(d::_GridVDF)
    T = promote_type(Float64, typeof(d.n))
    grids, f = map(x -> convert(Array, x), _grids(d)), convert(Array, d.f)
    all(≥(0), f) || throw(DomainError(minimum(f), "rand needs f ≥ 0"))
    # Each factor scaled to a maximum of 1, as products of SI-sized ones can underflow or overflow.
    w = map(_measure(typeof(d)), grids) do m, x
        wₖ = _interval_weights.(m, T.(@view x[1:(end - 1)]), T.(@view x[2:end]))
        s = maximum(t -> max(t...), wₖ)
        map(t -> t ./ s, wₖ)
    end
    W = T.(f) ./ T(maximum(f))
    for (k, wₖ) in enumerate(w)
        H = sum.(_halves.(Ref(wₖ), 1:(length(wₖ) + 1)))
        W .*= reshape(H, ntuple(j -> j == k ? length(H) : 1, ndims(W)))
    end
    return _GridSampler(d, grids, w, cumsum(vec(W)))
end

Random.rand(rng::AbstractRNG, d::_GridVDF) = rand(rng, _sampler(d))

function Random.rand(rng::AbstractRNG, s::_GridSampler)
    (; grids, w, cdf) = s
    # first node of cdf > u, skipping empty nodes
    n = CartesianIndices(map(length, grids))[min(searchsortedlast(cdf, rand(rng) * cdf[end]) + 1, length(cdf))]
    x = map(_measure(typeof(s.d)), grids, w, Tuple(n)) do m, g, wₖ, i
        l, r = _halves(wₖ, i)
        rand(rng) * (l + r) < l ? _draw_axis(rng, m, g[i - 1], g[i], true) : _draw_axis(rng, m, g[i], g[i + 1], false)
    end
    return _assemble(rng, s.d, x)
end

# x ∈ [a, b] of density ∝ φ(x) m(x), φ the hat of the right node, or the left one
function _draw_axis(rng, ::typeof(one), a::T, b::T, right) where {T}
    t = sqrt(rand(rng, T))
    return right ? a + (b - a) * t : b - (b - a) * t
end
# rejection from the hat alone, accepting ≥ 1/3 since 0 ≤ a < b
function _draw_axis(rng, ::typeof(_perp_measure), a::T, b::T, right) where {T}
    while true
        x = _draw_axis(rng, one, a, b, right)
        rand(rng, T) * b ≤ x && return x
    end
end

_assemble(rng, ::GyrotropicGridVDF, (v_perp, v_para)) = _with_gyrophase(rng, v_perp, v_para, typeof(v_perp))
_assemble(_, ::CartesianGridVDF, x) = SVector(x)

Base.show(io::IO, d::_GridVDF) = print(io, nameof(typeof(d)), "(", join(size(d.f), " × "), " grid, n = ", d.n, ")")
