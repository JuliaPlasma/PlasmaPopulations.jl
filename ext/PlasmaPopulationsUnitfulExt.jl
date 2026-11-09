module PlasmaPopulationsUnitfulExt

import PlasmaPopulations as PP
using Unitful: Unitful, AbstractQuantity, ustrip, uconvert, dimension, @u_str

PP.si_number_density(n::AbstractQuantity) = ustrip(u"m^-3", n)
PP.si_magnetic_field(B::AbstractQuantity) = ustrip(u"T", B)
PP.si_magnetic_field(B::AbstractArray{<:AbstractQuantity}) = ustrip.(u"T", B)
PP.si_velocity(v::AbstractQuantity) = ustrip(u"m/s", v)
PP.si_frequency(ω::AbstractQuantity) = ustrip(u"s^-1", ω)
PP.si_charge(q::AbstractQuantity) = ustrip(u"C", q)
PP.si_mass(m::AbstractQuantity) = ustrip(u"kg", m)
PP.si_phase_space_density(f::AbstractArray{<:AbstractQuantity}) = ustrip.(u"s^3/m^6", f)

# A temperature carries no energy dimension until multiplied by k_B
PP.si_energy(T::AbstractQuantity) =
    ustrip(u"J", dimension(T) == dimension(u"K") ? uconvert(u"J", T * Unitful.k) : T)

end
