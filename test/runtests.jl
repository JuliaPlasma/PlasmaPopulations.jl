using Aqua

include("particles-tests.jl")
include("distributions-tests.jl")

@testset "Aqua" begin
    Aqua.test_all(PlasmaPopulations; persistent_tasks = false)
end