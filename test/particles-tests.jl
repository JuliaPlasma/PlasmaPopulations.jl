using PlasmaPopulations, Test, Unitful
using PlasmaPopulations: E_SI, ME_SI, MP_SI, MU0_SI, U_SI, alfven_speed, debye_length, mass_density, gyrofrequency, plasma_frequency, thermal_speed

@testset "Particles" begin
    @test Particle("{1}H", 1).m ≈ MP_SI rtol = 1e-7  # H-atom binding energy
    @test Particle(:H, 1).m ≈ 1.008U_SI - ME_SI
    @test Particle(:Cl, 1).m ≈ 35.45U_SI - ME_SI  # natural abundance beyond the isotope table
    @test Particle("{18}O", 3).q == 3E_SI
    @test Particle("{18}O", 3).m ≈ 17.99915961286U_SI - 3ME_SI
    @test Particle(:NO, 1).m ≈ (14.007 + 15.999)U_SI - ME_SI
    @test Particle("{13}CO2", 1).m ≈ (13.00335483507 + 2 * 15.999)U_SI - ME_SI
    @test Particle(:D2, 1).m == Particle("{2}H2", 1).m ≈ 2 * 2.01410177784U_SI - ME_SI
    @test Particle("H{2}H", 0).m ≈ (1.008 + 2.01410177784)U_SI
    @test Particle(:O2, -1).m ≈ 2Particle(:O, 0).m + ME_SI
    @test_throws ArgumentError Particle(:H, 2)
    @test_throws ArgumentError Particle("{19}O", 1)
    @test_throws ArgumentError Particle("{2}D", 1)
    @test_throws ArgumentError Particle(:no, 1)
    @test_throws ArgumentError Particle(:Zz, 1)
    @test_throws ArgumentError Particle("O{18}", 1)
end

@testset "Unitful input" begin
    @test Particle(1.602176634e-19u"C", 1u"u").m ≈ ustrip(u"kg", 1u"u")
    s = Population(Proton(), Cold(); n = 5u"cm^-3")
    @test s.n ≈ 5e6
    @test thermal_speed(Electron(), 11604.518u"K") ≈ thermal_speed(Electron(), 1) rtol = 1e-6
    @test thermal_speed(Electron(), 1u"keV") ≈ thermal_speed(Electron(), 1000)
    @test Plasma(s; B0 = 5u"nT").B0 ≈ 5e-9
    @test Plasma(s, 5u"nT").populations == (s,)
    d = BiMaxwellian(; vth_perp = 1.0e3, vth_para = 1u"km/s", vd_para = 0.01u"c")
    @test d.vth_para == 1.0e3 && d.vd_para ≈ 0.01 * 2.99792458e8
    v = 1u"km/s"
    @test Cold(v).vd_para == 1.0e3
    @test GaussianRing(v, v, v, 0).vr_perp == 1.0e3
    @test BiKappa(v, v, 4, 0).vth_para == 1.0e3
    @test ProductBiKappa(v, v, 4, 4.0, 0, 0).vth_perp == 1.0e3
    @test_throws DomainError GaussianRing(-v, v, v, 0)
end

@testset "Formulary" begin
    p, e = Proton(), Electron()
    pl = Plasma(Population(p, Cold(); n = 1e6), Population(e, Cold(); n = 1e6); B0 = 1e-8)
    @test alfven_speed(pl) ≈ 1e-8 / sqrt(MU0_SI * 1e6 * (p.m + e.m))
    O = Population(Particle(:O, 0), Cold(); n = 1e12)
    @test alfven_speed(Plasma(pl.populations..., O; B0 = 1e-8)) == alfven_speed(pl)
    @test mass_density(Plasma(pl.populations..., O; B0 = 1e-8)) ≈ mass_density(pl) + 1e12 * O.particle.m
    @test gyrofrequency(e, 1e-8) < 0
    @test debye_length(e, 1e6, 10) ≈ thermal_speed(e, 10) / (sqrt(2) * plasma_frequency(e, 1e6))
end
