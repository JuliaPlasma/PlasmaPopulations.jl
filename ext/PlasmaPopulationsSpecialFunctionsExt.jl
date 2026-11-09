module PlasmaPopulationsSpecialFunctionsExt

import PlasmaPopulations as PP
using SpecialFunctions: erfc, besselkx

PP._erfc(x::Real) = erfc(x)
PP._besselkx(ν, x::Real) = besselkx(ν, x)

end
