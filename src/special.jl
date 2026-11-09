# `erfc` without a SpecialFunctions dependency: libm for hardware floats; the
# SpecialFunctions extension covers other number types.
erfc(x::Float64) = ccall((:erfc, Base.Math.libm), Float64, (Float64,), x)
erfc(x::Float32) = ccall((:erfcf, Base.Math.libm), Float32, (Float32,), x)
erfc(x::Real) = _erfc(float(x))
_erfc(x::Union{Float32, Float64}) = erfc(x)
function _erfc(x)
    throw(ArgumentError("erfc($(typeof(x))) needs `using SpecialFunctions`"))
end
_besselkx(ν, x) = throw(ArgumentError("besselkx needs `using SpecialFunctions`"))
