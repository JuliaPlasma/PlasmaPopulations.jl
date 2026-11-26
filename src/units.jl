# Input adapters, applied once at construction.
si_number_density(x) = x
si_magnetic_field(x) = x
si_velocity(x) = x
si_frequency(x) = x
si_charge(x) = x
si_mass(x) = x
si_energy(x) = x * oftype(float(x), E_SI)
si_phase_space_density(x) = x

_field_si(B::Number) = si_magnetic_field(B)
_field_si(B::Union{AbstractVector, Tuple}) = si_magnetic_field.(SVector{3}(B))
