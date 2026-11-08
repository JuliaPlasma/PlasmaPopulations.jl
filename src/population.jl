"""
    Population(particle, vdf; n)
    Population(particle, V::Type; n, kw...)

A population of `particle`s with number density `n` [m⁻³] and velocity distribution `vdf`.
"""
struct Population{P <: Particle, N, V}
    particle::P
    n::N
    vdf::V
    function Population(particle::Particle, n, vdf)
        n_si = si_number_density(n)
        return new{typeof(particle), typeof(n_si), typeof(vdf)}(particle, n_si, vdf)
    end
end

Population(p::Particle, vdf; n) = Population(p, n, vdf)
Population(p::Particle, V::Type; n, kw...) = Population(p, V(p; kw...); n)

"""
    Plasma(populations...; B0)
    Plasma(populations, B0)

Plasma state with fields `populations` and background `B0` [T].
"""
struct Plasma{S, B}
    populations::S
    B0::B
    function Plasma(populations, B0)
        B0_si = si_magnetic_field(B0)
        return new{typeof(populations), typeof(B0_si)}(populations, B0_si)
    end
end

Plasma(populations::Population...; B0) = Plasma(populations, B0)
Plasma(population::Population, B0) = Plasma((population,), B0)
