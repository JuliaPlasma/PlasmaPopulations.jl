module PlasmaPopulationsAdaptExt

using Adapt: Adapt
using PlasmaPopulations: GyrotropicGridVDF, CartesianGridVDF

Adapt.@adapt_structure GyrotropicGridVDF
Adapt.@adapt_structure CartesianGridVDF

end
