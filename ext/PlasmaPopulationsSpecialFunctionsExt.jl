module PlasmaPopulationsSpecialFunctionsExt

import PlasmaPopulations as PP
using SpecialFunctions: erfc

PP._erfc(x::Real) = erfc(x)

end
