using BenchmarkTools
using PlasmaPopulations
using PlasmaPopulations: pdf
using Random
using Statistics

const SUITE = BenchmarkGroup()

const rng = Xoshiro(1)
const models = (
    BiMaxwellian = BiMaxwellian(; vth_perp = 1.3, vth_para = 0.7, vd_para = 0.4),
    BiKappa = BiKappa(; vth_perp = 1.3, vth_para = 0.7, kappa = 3.5),
    MaxwellJuttner = MaxwellJuttner(; theta = 0.5),
)

SUITE["pdf"] = BenchmarkGroup()
SUITE["rand"] = BenchmarkGroup()
for (name, d) in pairs(models)
    SUITE["pdf"][string(name)] = @benchmarkable pdf($d, 1.1, 0.3)
    SUITE["rand"][string(name)] = @benchmarkable rand($rng, $d)
end
const oriented = Oriented(models.BiKappa, [1.0, 2.0, 3.0]; u = [0.1, 0.0, 0.0])
SUITE["pdf"]["Oriented"] = @benchmarkable pdf($oriented, $([0.5, -0.2, 1.0]))

const vperp, vpara = range(0, 8, 321), range(-5, 5, 401)
const F2 = [pdf(models.BiMaxwellian, v, u) for v in vperp, u in vpara]
const gyro = GyrotropicGridVDF(vperp, vpara, F2)
const vx, vy, vz = range(-6, 6, 97), range(-6, 6, 89), range(-6, 6, 81)
const F3 = [pdf(oriented, [a, b, c]) for a in vx, b in vy, c in vz]
const cart = CartesianGridVDF(vx, vy, vz, F3)

SUITE["grid"] = BenchmarkGroup()
SUITE["grid"]["GyrotropicGridVDF"] = @benchmarkable GyrotropicGridVDF($vperp, $vpara, $F2)
SUITE["grid"]["GyrotropicGridVDF vth_perp_para"] = @benchmarkable vth_perp_para($gyro)
SUITE["grid"]["GyrotropicGridVDF pdf"] = @benchmarkable pdf($gyro, 1.1, 0.3)
SUITE["grid"]["CartesianGridVDF"] = @benchmarkable CartesianGridVDF($vx, $vy, $vz, $F3)
SUITE["grid"]["CartesianGridVDF cov"] = @benchmarkable cov($cart)
SUITE["grid"]["CartesianGridVDF pdf"] = @benchmarkable pdf($cart, $([0.5, -0.2, 1.0]))
