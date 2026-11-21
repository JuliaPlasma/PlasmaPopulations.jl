module PlasmaPopulations

using Random: AbstractRNG, randexp, randn, rand
import Random
using Bessels: besselkx
using Gamma: loggamma
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm, tr
import Statistics
using Statistics: mean, cov

export Particle, Electron, Positron, Proton
export AbstractVDF, BiMaxwellian, GaussianRing, BiKappa, ProductBiKappa, DoryGuestHarris, SubtractedMaxwellian, Cold, MaxwellJuttner, GridVDF, vth_perp_para, isgyrotropic
export Population, Plasma
export Oriented

VERSION >= v"1.11.0-DEV.469" && eval(Meta.parse("""public pdf, gyrofrequency, plasma_frequency, thermal_speed, inertial_length, debye_length, alfven_speed, mass_density,
    temperature_perp_para, plasma_beta, agyrotropy, charge_density, current_density,
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

debye_length(p::Particle, n, T) = sqrt(EPS0_SI * si_energy(T) / (abs(si_number_density(n)) * p.q^2))
gyrofrequency(p::Particle, 𝐁) = p.q * norm(si_magnetic_field(𝐁)) / p.m
plasma_frequency(p::Particle, n) = abs(p.q) * sqrt(abs(si_number_density(n)) / (EPS0_SI * p.m))
thermal_speed(p::Particle, T) = sqrt(2 * si_energy(T) / p.m)
inertial_length(p::Particle, n) = C_SI / plasma_frequency(p, n)

mass_density(plasma::Plasma) = sum(s -> s.n * s.particle.m, plasma.populations)
# Charged populations alone: neutrals load Alfvén waves only below the ion–neutral collision frequency, which a Plasma does not carry.
alfven_speed(plasma::Plasma) =
    norm(plasma.B0) / sqrt(MU0_SI * sum(s -> !iszero(s.particle.q) * s.n * s.particle.m, plasma.populations))

temperature_perp_para(s::Population) = s.particle.m .* _temperature_per_mass(s.vdf) ./ E_SI
plasma_beta(s::Population, B) = (2 * MU0_SI * s.n * s.particle.m / norm(si_magnetic_field(B))^2) .* _temperature_per_mass(s.vdf)
plasma_beta(plasma::Plasma) =
    mapreduce(s -> !iszero(s.particle.q) .* plasma_beta(s, plasma.B0), (a, b) -> a .+ b, plasma.populations)

# Swisdak (2016) √Q in rotation invariants.
function agyrotropy(P, B)
    b = B / norm(B)
    Ppara = b' * P * b
    I₁ = tr(P)
    I₂ = P[1, 1] * P[2, 2] + P[1, 1] * P[3, 3] + P[2, 2] * P[3, 3] - P[1, 2] * P[2, 1] - P[1, 3] * P[3, 1] - P[2, 3] * P[3, 2]
    return sqrt(max(1 - 4I₂ / ((I₁ - Ppara) * (I₁ + 3Ppara)), 0))
end

charge_density(plasma::Plasma) = sum(s -> s.n * s.particle.q, plasma.populations)

current_density(plasma::Plasma) = sum(s -> s.n * s.particle.q * mean(s.vdf), plasma.populations)


end
