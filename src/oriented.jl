"""
    Oriented(d, axis; u)
    Oriented(d; u)

Model `d` turned by the shortest rotation from `ẑ` to the lab-frame vector `axis`
(π about `x̂` if `axis ∥ -ẑ`), then shifted by the lab velocity `u` (default zero).
"""
struct Oriented{D, R, U} <: AbstractVDF
    vdf::D
    basis::R
    u::U
end

# false: Bool is the weakest promoting Number, so the model's float type wins.
Oriented(d, axis; u = SVector(false, false, false)) =
    Oriented(d, _field_aligned_basis(SVector{3}(axis), float(eltype(mean(d)))), si_velocity.(SVector{3}(u)))
Oriented(d; u = SVector(false, false, false)) = Oriented(d, I, si_velocity.(SVector{3}(u)))

# Columns (e⊥₁, e⊥₂, b̂) of the shortest rotation ẑ → b̂: R = I + [ẑ × b̂]× + [ẑ × b̂]×² / (1 + b₃).
# For b₃ < 0, 1 + b₃ = ρ²/(|B|(|B| - B₃)) avoids the cancellation that would cost eps/(1 + b₃).
# `* one(T)` promotes so integer B takes the model's float type T
function _field_aligned_basis(B, ::Type{T}) where {T}
    iszero(B) && throw(ArgumentError("B = $B has no direction"))
    B = B * one(T)
    n = norm(B)
    b = B / n
    ρ² = B[1]^2 + B[2]^2
    o, z = one(n), zero(n)
    b[3] < 0 && iszero(ρ²) && return hcat(SVector(o, z, z), SVector(z, -o, z), b)
    k = b[3] < 0 ? n * (n - B[3]) / ρ² : 1 / (1 + b[3])
    c = -b[1] * b[2] * k
    return hcat(SVector(1 - b[1]^2 * k, c, -b[1]), SVector(c, 1 - b[2]^2 * k, -b[2]), b)
end

pdf(o::Oriented) = _PDF(o, pdf(o.vdf))
(f::_PDF{<:Oriented})(v) = f.c(f.d.basis' * (si_velocity.(SVector{3}(v)) - f.d.u))
pdf(o::Oriented, v) = pdf(o)(v)

Statistics.mean(o::Oriented) = o.u + o.basis * mean(o.vdf)
Statistics.cov(o::Oriented) = o.basis * cov(o.vdf) * o.basis'
vth_perp_para(o::Oriented) = vth_perp_para(o.vdf)
_temperature_per_mass(o::Oriented) = _temperature_per_mass(o.vdf)

Random.rand(rng::AbstractRNG, o::Oriented) = o.u + o.basis * rand(rng, o.vdf)
