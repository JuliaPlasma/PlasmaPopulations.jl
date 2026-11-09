module PlasmaPopulations

using Random: AbstractRNG, randexp, randn, rand
import Random
using Gamma: loggamma
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm
import Statistics
using Statistics: mean, cov

export Particle, Electron, Positron, Proton
export AbstractVDF, BiMaxwellian, GaussianRing, BiKappa, ProductBiKappa, Cold, MaxwellJuttner, GridVDF, vth_perp_para, isgyrotropic
export Population, Plasma
export Oriented

VERSION >= v"1.11.0-DEV.469" && eval(Meta.parse("""public pdf, gyrofrequency, plasma_frequency, thermal_speed, inertial_length, debye_length, alfven_speed, mass_density,
    kappa_widths,
    si_number_density, si_magnetic_field, si_velocity, si_frequency, si_charge, si_mass, si_energy, si_phase_space_density,
    C_SI, E_SI, ME_SI, MP_SI, U_SI, EPS0_SI, MU0_SI, KB_SI"""))

include("constants.jl")
include("units.jl")
include("special.jl")
include("nuclides.jl")
include("particles.jl")
include("distributions.jl")
include("grid.jl")
include("oriented.jl")
include("population.jl")

# formulary
debye_length(p::Particle, n, T) = sqrt(EPS0_SI * si_energy(T) / (abs(si_number_density(n)) * p.q^2))
gyrofrequency(p::Particle, 𝐁) = p.q * norm(si_magnetic_field(𝐁)) / p.m
plasma_frequency(p::Particle, n) = abs(p.q) * sqrt(abs(si_number_density(n)) / (EPS0_SI * p.m))
thermal_speed(p::Particle, T) = sqrt(2 * si_energy(T) / p.m)
inertial_length(p::Particle, n) = C_SI / plasma_frequency(p, n)

mass_density(plasma::Plasma) = sum(s -> s.n * s.particle.m, plasma.populations)
# Charged populations alone: neutrals load Alfvén waves only below the ion–neutral collision frequency, which a Plasma does not carry.
alfven_speed(plasma::Plasma) =
    norm(plasma.B0) / sqrt(MU0_SI * sum(s -> !iszero(s.particle.q) * s.n * s.particle.m, plasma.populations))


end
