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
function _field_aligned_basis(B, ::Type{T}) where {T}
    B = _field_axis(B, T)
    b = B / norm(B)
    x, y, z = b
    if z < 0
        # (1 - z) times the transverse unit vector's outer product stays bounded near -ẑ.
        ρ = hypot(x, y)
        if iszero(ρ)
            o, t = one(z), zero(z)
            return hcat(SVector(o, t, t), SVector(t, -o, t), b)
        end
        x, y, k = x / ρ, y / ρ, 1 - z
    else
        k = 1 / (1 + z)
    end
    c = -x * y * k
    return hcat(SVector(1 - x^2 * k, c, -b[1]), SVector(c, 1 - y^2 * k, -b[2]), b)
end

# Unnormalized (⊥, ∥) axis of field `B` in float type `T` (`Bool`: no promotion): the vector, or lab `ẑ` for a scalar |B|.
function _field_axis(B, ::Type{T} = Bool) where {T}
    iszero(B) && throw(DomainError(B, "a zero field or axis has no direction"))
    return B isa Number ? _along_z(one(T)) : B * one(T)
end

# Unlike (⊥, ∥) splits, zero field is allowed: drifts then stay in the model's coordinates.
function _mean_in_field(d, B)
    μ = mean(d)
    (iszero(B) || !isgyrotropic(d)) && return μ
    return μ[3] * normalize(_field_axis(B, float(eltype(μ))))
end

# Rotate in the model frame: rotating an anisotropic tensor into the lab first can erase its small variance.
_rotated_tensor(tensor_of, o::Oriented, axis) = _rotated_tensor(tensor_of, o.vdf, o.basis' * SVector{3}(axis))
function _rotated_tensor(tensor_of, d, axis)
    M = tensor_of(d)
    R = _field_aligned_basis(SVector{3}(axis), eltype(M))
    return R' * M * R
end

function vth_perp_para(d::AbstractVDF, B)
    axis = _field_axis(_field_si(B))
    return isgyrotropic(d) ? vth_perp_para(d) : _vth_perp_para(_rotated_tensor(cov, d, axis))
end

function _temperature_per_mass(d, B_si)
    axis = _field_axis(B_si)
    return isgyrotropic(d) ? _temperature_per_mass(d) : _perp_para(_rotated_tensor(_temperature_tensor, d, axis))
end

pdf(o::Oriented) = _PDF(o, pdf(o.vdf))
(f::_PDF{<:Oriented})(v) = f.c(f.d.basis' * (si_velocity.(SVector{3}(v)) - f.d.u))
pdf(o::Oriented, v) = pdf(o)(v)

Statistics.mean(o::Oriented) = o.u + o.basis * mean(o.vdf)
Statistics.cov(o::Oriented) = o.basis * cov(o.vdf) * o.basis'

Random.rand(rng::AbstractRNG, o::Oriented) = o.u + o.basis * rand(rng, o.vdf)
