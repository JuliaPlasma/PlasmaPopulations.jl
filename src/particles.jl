struct Particle{T <: Real}
    q::T
    m::T
    Particle(q, m) = (qm = promote(si_charge(q), si_mass(m)); new{eltype(qm)}(qm...))
end

Base.broadcastable(p::Particle) = Ref(p)
Base.show(io::IO, p::Particle) = print(io, "Particle(z = ", p.q / E_SI, ", m = ", p.m / U_SI, " u)")

Electron() = Particle(-E_SI, ME_SI)
Positron() = Particle(E_SI, ME_SI)
Proton() = Particle(E_SI, MP_SI)

"""
    Particle(formula, z)

Mass is the sum of neutral atomic masses less `z` electrons; binding energies are neglected.
"""
function Particle(formula::Union{Symbol, AbstractString}, z::Integer)
    m, Z = 0.0, 0
    for (Zi, mi, count) in _parse_formula(String(formula))
        m += count * mi
        Z += count * Zi
    end
    z ≤ Z || throw(ArgumentError("$formula has only $Z electrons to strip, got z = $z"))
    return Particle(z * E_SI, m * U_SI - z * ME_SI)
end

# (atomic number, mass [u], count) per segment
function _parse_formula(str::String)
    atoms = Tuple{Int, Float64, Int}[]
    pos = 1
    for m in eachmatch(r"(?:\{(\d+)\})?([A-Z][a-z]?)(\d*)", str)
        m.offset == pos || break
        el = Symbol(m[2])
        el, A = isnothing(m[1]) ? get(ISOTOPE_SYMBOLS, el, (el, nothing)) : (el, parse(Int, m[1]))
        haskey(ELEMENTS, el) || throw(ArgumentError("unknown element $el in $str"))
        Z, weight = ELEMENTS[el]
        mass = isnothing(A) ? weight : get(ISOTOPES, (el, A)) do
            known = sort!([k[2] for k in keys(ISOTOPES) if k[1] == el])
            throw(ArgumentError("no mass for {$A}$el; known: $known; use Particle(q, m)"))
        end
        push!(atoms, (Z, mass, isempty(m[3]) ? 1 : parse(Int, m[3])))
        pos += ncodeunits(m.match)
    end
    (pos == ncodeunits(str) + 1 && !isempty(atoms)) || throw(ArgumentError("cannot parse formula $str"))
    return atoms
end
