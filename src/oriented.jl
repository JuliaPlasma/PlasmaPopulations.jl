"""
    Oriented(d, B; u = 0)

Field-aligned model `d` placed in the frame of magnetic field `B` and bulk velocity `u`.
"""
struct Oriented{D, R, U} <: AbstractVDF
    vdf::D
    basis::R
    u::U
end

# false: Bool is the weakest promoting Number, so the model's float type wins.
Oriented(d, B; u = SVector(false, false, false)) =
    Oriented(d, _field_aligned_basis(SVector{3}(B)), si_velocity.(SVector{3}(u)))

# Columns (e⊥₁, e⊥₂, b̂), by Duff et al. (2017): a rotation of ẑ alone is singular at b̂ = -ẑ.
function _field_aligned_basis(B)
    b = B / norm(B)
    s = copysign(one(b[3]), b[3])
    a = -1 / (s + b[3])
    c = b[1] * b[2] * a
    return hcat(SVector(1 + s * b[1]^2 * a, s * c, -s * b[1]), SVector(c, s + b[2]^2 * a, -b[2]), b)
end

pdf(o::Oriented, v) = pdf(o.vdf, o.basis' * (si_velocity.(SVector{3}(v)) - o.u))

Statistics.mean(o::Oriented) = o.u + o.basis * mean(o.vdf)
Statistics.cov(o::Oriented) = o.basis * cov(o.vdf) * o.basis'
vth_perp_para(o::Oriented) = vth_perp_para(o.vdf)
_temperature_per_mass(o::Oriented) = _temperature_per_mass(o.vdf)

Random.rand(rng::AbstractRNG, o::Oriented) = o.u + o.basis * rand(rng, o.vdf)
