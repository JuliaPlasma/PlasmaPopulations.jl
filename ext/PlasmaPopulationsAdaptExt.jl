module PlasmaPopulationsAdaptExt

using Adapt: Adapt
using PlasmaPopulations: GyrotropicGridVDF

Adapt.@adapt_structure GyrotropicGridVDF

end
