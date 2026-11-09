using PlasmaPopulations, Test
using LinearAlgebra, QuadGK, Random, SpecialFunctions, Statistics, Unitful
using PlasmaPopulations: pdf, alfven_speed, gyrofrequency, debye_length, kappa_widths, mass_density, plasma_frequency, thermal_speed,
    temperature_perp_para, plasma_beta, charge_density, current_density, E_SI, MU0_SI

@testset "Moments of pdf" begin
    # ∫ f d³v = 1, ⟨v∥⟩ = vd_para, ⟨v⊥²⟩ = vth⊥², ⟨(v∥-vd_para)²⟩ = vth∥²/2
    function moments(d; vmax)
        I(g) = quadgk(v_perp -> 2π * v_perp * quadgk(v_para -> g(v_perp, v_para) * pdf(d, v_perp, v_para), -vmax, vmax; rtol = 1e-10)[1], 0, vmax; rtol = 1e-10)[1]
        # ⟨v∥⟩ is shifted by 1 because quadgk's rtol cannot converge on a zero integral.
        return I((a, b) -> 1), I((a, b) -> b + 1) - 1, I((a, b) -> a^2), I((a, b) -> (b - d.vd_para)^2)
    end
    vth = (vth_perp = 1.3, vth_para = 0.7)
    for d in (GaussianRing(; vr_perp = 2.0, w_perp = 1.3, vth_para = 0.7, vd_para = 0.4), BiMaxwellian(; vth..., vd_para = 0.4), BiKappa(; vth..., kappa = 6.0, vd_para = -0.2), ProductBiKappa(; vth..., kappa_perp = 5.0, kappa_para = 4.0, vd_para = 0.1),
            ProductBiKappa(; vth..., kappa_perp = 5.0, kappa_para = 4.0, loss_cone = 1.5),
            DoryGuestHarris(; vth..., loss_cone = 1.5, vd_para = 0.4), SubtractedMaxwellian(; vth..., delta = 0.3, beta = 0.4, vd_para = 0.4))
        n, vpara, vperp2, vpara2 = moments(d; vmax = d isa Union{BiKappa, ProductBiKappa} ? 400.0 : 15.0)
        vth_perp, vth_para = vth_perp_para(d)
        @test n ≈ 1 rtol = 1e-6
        @test vpara ≈ mean(d)[3] atol = 1e-3
        @test vperp2 ≈ vth_perp^2 rtol = 1e-3
        @test vpara2 ≈ vth_para^2 / 2 rtol = 1e-3
        @test pdf(d, -0.8, 0.3) == pdf(d, 0.8, 0.3)
    end
    @test vth_perp_para(Cold(; vd_para = 1.0)) == (0.0, 0.0)
    @test mean(Cold(; vd_para = 1.0)) == [0, 0, 1]
    @test vth_perp_para(GaussianRing(; vr_perp = 2.0, w_perp = 1.0e-6, vth_para = 1.0))[1] ≈ 2
    @test vth_perp_para(GaussianRing(; vr_perp = 0.0, w_perp = 1.3, vth_para = 0.7)) == vth_perp_para(BiMaxwellian(; vth...))
    @test_throws DomainError GaussianRing(; vr_perp = 1.0, w_perp = 0.0, vth_para = 1.0)
end

@testset "vth sets both thermal speeds, giving isotropy" begin
    for d in (BiMaxwellian(; vth = 2.0), BiKappa(; vth = 2.0, kappa = 4))
        @test pdf(d, 0.6, 0.8) ≈ pdf(d, 0.8, 0.6) ≈ pdf(d, 1.0, 0.0)
    end
    @test ProductBiKappa(; vth = 2.0, kappa_perp = 4, kappa_para = 5) == ProductBiKappa(; vth_perp = 2.0, vth_para = 2.0, kappa_perp = 4, kappa_para = 5)
    @test_throws ArgumentError BiMaxwellian(; vth_perp = 2.0)
    @test_throws ArgumentError BiKappa(; kappa = 4)
end

@testset "pdf takes velocity vectors and units" begin
    d = BiMaxwellian(; vth_perp = 1.3, vth_para = 0.7, vd_para = 0.4)
    @test pdf(d, (3.0, 4.0, 0.4)) == pdf(d, 5.0, 0.4)
    @test pdf(d, 1u"km/s", 2u"m/s") == pdf(d, 1000.0, 2.0)
    @test pdf(d, [3u"km/s", 4u"km/s", 400.0u"m/s"]) ≈ pdf(d, 5000.0, 400.0)
end

@testset "Oriented places a model in the frame of B and u" begin
    d = BiMaxwellian(; vth_perp = 2.0, vth_para = 1.0, vd_para = 0.5)
    u = [10.0, -1.0, 2.0]
    for B in ([0, 0, 2.0], [0, 0, -1.0], [1.0, 0, 0], [1.0, 2.0, -3.0])
        b = normalize(B)
        o = Oriented(d, B; u)
        v = [0.3, -1.1, 0.7]
        v_para = v ⋅ b
        @test pdf(o, v + u) ≈ pdf(d, norm(v - v_para * b), v_para)
        @test mean(o) ≈ u + d.vd_para * b
        xs = rand(Xoshiro(1), o, 200_000)
        @test mean(xs) ≈ mean(o) atol = 0.02
        @test cov(reduce(hcat, xs); dims = 2) ≈ cov(o) rtol = 0.02
        @test vth_perp_para(o) == vth_perp_para(d)
    end
    o = Oriented(d, [0.0, 0.0, 5.0]u"nT"; u = [1.0, 0.0, 0.0]u"km/s")
    @test pdf(o, [1000.0, 0.0, 0.5]u"m/s") ≈ pdf(d, 0.0, 0.5)
    @test size(rand(o, 2, 3)) == (2, 3)
    @test eltype(rand(Oriented(BiMaxwellian(; vth = 1.0f0), Float32[0, 0, 1]))) === Float32
end

@testset "B0 is a magnitude or a vector" begin
    p = Proton()
    s = Population(p, Cold(); n = 1.0e6)
    for B0 in (-5.0e-9, [0.0, 3.0e-9, 4.0e-9], [0.0, 3.0, 4.0]u"nT")
        @test alfven_speed(Plasma(s; B0)) ≈ alfven_speed(Plasma(s; B0 = 5.0e-9))
        @test gyrofrequency(p, B0) ≈ gyrofrequency(p, 5.0e-9)
    end
end

@testset "negative density" begin
    p = Proton()
    @test plasma_frequency(p, -4.0e6) == plasma_frequency(p, 4.0e6)
    @test debye_length(p, -4.0e6, 10.0) == debye_length(p, 4.0e6, 10.0)
    plasma = Plasma(Population(p, Cold(); n = 3.0e6), Population(p, Cold(); n = -1.0e6); B0 = 1.0e-8)
    @test mass_density(plasma) ≈ 2.0e6 * p.m
end

@testset "Kappa limits" begin
    vth = (vth_perp = 1.0, vth_para = 2.0)
    m = BiMaxwellian(; vth...)
    @test pdf(BiKappa(; vth..., kappa = 1e5), 0.7, 0.3) ≈ pdf(m, 0.7, 0.3) rtol = 1e-3
    @test pdf(ProductBiKappa(; vth..., kappa_perp = 1e5, kappa_para = 1e5), 0.7, 0.3) ≈ pdf(m, 0.7, 0.3) rtol = 1e-3
    k = BiKappa(; vth..., kappa = 4.0)
    θperp, θpara = kappa_widths(k)
    @test pdf(k, 0.7, 0.3) / pdf(k, 0.0, 0.0) ≈ (1 + 0.7^2 / (4θperp^2) + 0.3^2 / (4θpara^2))^-5
    @test_throws DomainError BiKappa(; vth..., kappa = 1.5)
    @test_throws DomainError ProductBiKappa(; vth..., kappa_perp = 1.0, kappa_para = 2.0)

    lc = ProductBiKappa(; vth..., kappa_perp = 1e5, kappa_para = 1e5, loss_cone = 2)
    @test pdf(lc, 0.7, 0.3) ≈ pdf(DoryGuestHarris(; vth..., loss_cone = 2), 0.7, 0.3) rtol = 1e-3
    @test pdf(lc, 0.0, 0.3) == 0
    @test pdf(DoryGuestHarris(; vth..., loss_cone = 0), 0.7, 0.3) ≈ pdf(m, 0.7, 0.3)
    @test pdf(SubtractedMaxwellian(; vth..., delta = 1, beta = 0.5), 0.7, 0.3) ≈ pdf(m, 0.7, 0.3)
    @test pdf(SubtractedMaxwellian(; vth..., delta = 0, beta = 0.5), 0.0, 0.3) == 0
    @test_throws DomainError SubtractedMaxwellian(; vth..., delta = 0.5, beta = 0)
end

@testset "Samples follow pdf" begin
    rng = Xoshiro(1)
    vth = (vth_perp = 1.0, vth_para = 2.0)
    ring(vr_perp, w_perp) = GaussianRing(; vr_perp, w_perp, vth_para = 2.0, vd_para = 0.5)
    # the rejection sampler's mixture weights depend on vr_perp/w_perp
    for d in (BiMaxwellian(; vth..., vd_para = 0.5), BiKappa(; vth..., kappa = 3.0, vd_para = 0.5),
            ProductBiKappa(; vth..., kappa_perp = 3.0, kappa_para = 3.0, loss_cone = 2, vd_para = 0.5),
            DoryGuestHarris(; vth..., loss_cone = 1.5, vd_para = 0.5), SubtractedMaxwellian(; vth..., delta = 0.3, beta = 0.4, vd_para = 0.5),
            ring(0.0, 1.0), ring(0.3, 1.0), ring(2.0, 1.0), ring(5.0, 0.5))
        vs = rand(rng, d, 200_000)
        vperp0, vpara0 = vth_perp_para(d)
        @test mean(vs) ≈ mean(d) atol = 5 * hypot(vperp0, vpara0) / sqrt(length(vs))
        # Joint probability: the coupled models' ⊥ and ∥ are dependent, so marginals would not see it.
        for (a, b) in ((0.7, 0.5), (1.2, 1.5))
            P = quadgk(v -> 2π * v * quadgk(u -> pdf(d, v, u), d.vd_para - b * vpara0, d.vd_para + b * vpara0)[1], 0, a * vperp0)[1]
            @test mean(v -> hypot(v[1], v[2]) < a * vperp0 && abs(v[3] - d.vd_para) < b * vpara0, vs) ≈ P atol = 0.005
        end
    end
end

@testset "Models keep the speeds' float type and precision" begin
    for T in (Float32, BigFloat)
        vth, v = one(T), T(1) / 2
        for d in (BiMaxwellian(; vth), GaussianRing(; vr_perp = vth, w_perp = vth, vth_para = vth), BiKappa(; vth, kappa = 4),
                ProductBiKappa(; vth, kappa_perp = 4, kappa_para = 4, loss_cone = 1), DoryGuestHarris(; vth, loss_cone = 1),
                SubtractedMaxwellian(; vth, delta = 1 // 2, beta = 1 // 4))
            @test pdf(d, v, v) isa T
            @test eltype(rand(d)) === T
        end
    end
    setprecision(256) do
        v = big"0.5"
        @test abs(pdf(BiMaxwellian(; vth = big"1.0"), v, v) - exp(-2v^2) / big(π)^big"1.5") < big"1e-70"
        A = big"2.5"   # κ - 3/2, with Γ(5)/Γ(7/2) = 64/(5√π)
        exact = 64 / (5sqrt(big(π))) / (big(π) * A)^big"1.5" * (1 + 2v^2 / A)^(-5)
        @test abs(pdf(BiKappa(; vth = big"1.0", kappa = big"4.0"), v, v) - exact) < big"1e-70"
    end
end

@testset "erfc without SpecialFunctions; other types through its extension" begin
    @test PlasmaPopulations.erfc(0.5) ≈ 0.4795001221869535
    @test PlasmaPopulations.erfc(big"0.5") ≈ SpecialFunctions.erfc(big"0.5")
end

@testset "Population builds a model from temperatures or speeds" begin
    p = Proton()
    Tperp, Tpara = 100u"eV", 20u"eV"
    v_perp, v_para = thermal_speed(p, Tperp), thermal_speed(p, Tpara)

    s = Population(p, ProductBiKappa; T_perp = Tperp, T_para = Tpara, kappa_perp = 5.5, kappa_para = 4.5, vd_para = 0.01u"c", n = 5u"cm^-3")
    @test s.vdf == ProductBiKappa(; vth_perp = v_perp, vth_para = v_para, kappa_perp = 5.5, kappa_para = 4.5, vd_para = 0.01u"c")
    @test s.n ≈ 5.0e6
    @test BiMaxwellian(p; T = 50) == BiMaxwellian(; vth = thermal_speed(p, 50))
    @test Population(p, BiMaxwellian; vth = 1.0u"km/s", n = 1.0).vdf == BiMaxwellian(; vth = 1.0e3)
    @test Population(p, Cold; vd_para = 1u"km/s", n = 1.0).vdf.vd_para == 1.0e3
    @test Population(p, GaussianRing; vr_perp = 1.0, w_perp = 2.0, T_para = Tpara, n = 1.0).vdf == GaussianRing(; vr_perp = 1.0, w_perp = 2.0, vth_para = v_para)
    @test Population(p, GaussianRing; vr_perp = 1.0, w_perp = 2.0, vth_para = 3.0, n = 1.0).vdf == GaussianRing(; vr_perp = 1.0, w_perp = 2.0, vth_para = 3.0)

    @test_throws ArgumentError BiMaxwellian(p; T_perp = Tperp)
end

@testset "Temperatures, β, charge and current of a plasma" begin
    p, e = Proton(), Electron()
    ions = Population(p, BiMaxwellian; T_perp = 100, T_para = 20, n = 5.0e6)
    plasma = Plasma(ions, Population(e, BiMaxwellian; T = 50, vd_para = 1.0e5, n = 5.0e6); B0 = 5.0e-9)
    @test all(temperature_perp_para(ions) .≈ (100, 20))
    @test all(temperature_perp_para(Population(e, MaxwellJuttner; T = 511.0e3, n = 1.0)) .≈ 511.0e3)
    @test all(plasma_beta(ions, plasma.B0) .≈ 2MU0_SI * 5.0e6 * E_SI .* (100, 20) ./ 25.0e-18)
    @test all(plasma_beta(plasma) .≈ 2MU0_SI * 5.0e6 * E_SI .* (150, 70) ./ 25.0e-18)
    @test charge_density(plasma) == 0
    @test current_density(plasma) ≈ [0, 0, -5.0e6 * E_SI * 1.0e5]
end

@testset "GridVDF tabulates a gyrotropic distribution" begin
    m = BiMaxwellian(; vth_perp = 1.3, vth_para = 0.7, vd_para = 0.4)
    v_perp, v_para = range(0, 8, 321), range(-5, 5, 401)
    F = [3.0 * pdf(m, v, u) for v in v_perp, u in v_para]
    g = GridVDF(v_perp, v_para, F)
    @test g.n ≈ 3 rtol = 1e-3
    @test mean(g) ≈ mean(m) atol = 1e-4
    @test all(isapprox.(vth_perp_para(g), vth_perp_para(m); rtol = 1e-3))
    @test pdf(g, v_perp[5], v_para[7]) ≈ F[5, 7] / g.n
    @test pdf(g, -(v_perp[5] + v_perp[6]) / 2, v_para[7]) ≈ (F[5, 7] + F[6, 7]) / 2g.n
    @test pdf(g, 1.0, 6.0) == 0
    # density of the interpolant: exact for f linear in v⊥ and v∥
    lin = GridVDF([0.0, 2.0], [-1.0, 1.0], [1.0 1.0; 3.0 3.0])
    @test lin.n ≈ 2π * 2 * (2 + 8 / 3)
    @test GridVDF(v_perp .* u"m/s", v_para .* u"m/s", F .* u"s^3/cm^6").n ≈ 3e12 rtol = 1e-3
    @test_throws DimensionMismatch GridVDF(v_perp, v_para, F')
    @test_throws ArgumentError GridVDF(reverse(v_perp), v_para, F)
    @test_throws DomainError GridVDF(v_perp .- 1, v_para, F)
end

@testset "MaxwellJuttner" begin
    c = PlasmaPopulations.C_SI
    radial(d, g) = quadgk(u -> 4π * u^2 * g(u) * pdf(d, u, 0.0), 0, Inf; rtol = 1e-10)[1]
    for θ in (1e-3, 0.5, 5.0)
        d = MaxwellJuttner(; theta = θ)
        @test radial(d, one) ≈ 1 rtol = 1e-8
        @test radial(d, abs2) ≈ sum(abs2, vth_perp_para(d)) * 3 / 4 rtol = 1e-8   # ⟨u²⟩ = 3⟨u_z²⟩
    end
    nr = MaxwellJuttner(; theta = 1e-6)
    @test pdf(nr, 0.3c * 1e-3, 0.2c * 1e-3) ≈ pdf(BiMaxwellian(; vth = c * sqrt(2e-6)), 0.3c * 1e-3, 0.2c * 1e-3) rtol = 1e-4
    @test MaxwellJuttner(Electron(); T = 511u"keV").theta ≈ 1 rtol = 1e-3
    @test_throws DomainError MaxwellJuttner(; theta = 0)
end
