using Aqua

include("particles-tests.jl")
include("distributions-tests.jl")

@testset "Aqua" begin
    Aqua.test_all(PlasmaPopulations)
end