using PlasmaPopulations, Test
using JLArrays, LinearAlgebra, QuadGK, Random, SpecialFunctions, Statistics, Unitful
using PlasmaPopulations: pdf, alfven_speed, gyrofrequency, debye_length, kappa_widths, mass_density, plasma_frequency, thermal_speed,
    temperature_perp_para, plasma_beta, agyrotropy, charge_density, current_density, E_SI, MU0_SI

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
    beam = BiMaxwellian(; vth_perp = Int32(100_000), vth_para = Int32(50_000))
    @test vth_perp_para(beam, [3, 0, 4]u"nT") == vth_perp_para(beam)
    @test diag(cov(beam)) == [5e9, 5e9, 1.25e9]
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
    @test isbits(pdf(Oriented(GaussianRing(; vr_perp = 2.0, w_perp = 1.3, vth_para = 0.7), (1, 0, 0))))
    @test pdf(d).([(3.0, 4.0, 0.4)]) == [pdf(d, 5.0, 0.4)]
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
        @test all(vth_perp_para(o, B) .≈ vth_perp_para(d))
    end
    o = Oriented(d, [0.0, 0.0, 5.0]u"nT"; u = [1.0, 0.0, 0.0]u"km/s")
    @test pdf(o, [1000.0, 0.0, 0.5]u"m/s") ≈ pdf(d, 0.0, 0.5)
    @test size(rand(o, 2, 3)) == (2, 3)
    @test eltype(rand(Oriented(BiMaxwellian(; vth = 1.0f0), Float32[0, 0, 1]))) === Float32
    @test pdf(Oriented(BiMaxwellian(; vth = 1.0f0), (1, 0, 0)), (0.1f0, 0.2f0, 0.3f0)) isa Float32
    @test_throws DomainError Oriented(d, (0, 0, 0))
    # A transverse drift tells rotations about the axis apart: ẑ → x̂ is about ŷ on both sides of the equator.
    drifting = Oriented(Cold(); u = (1, 2, 3))
    for z in (1.0e-9, -0.0, -1.0e-9)
        @test mean(Oriented(drifting, (1, 0, z))) ≈ [3, 2, -1]
    end
    @test mean(Oriented(drifting, (0, 0, -1))) ≈ [1, -2, -3]
    R = Oriented(BiMaxwellian(; vth = 1.0f0), Float32[1.0e-3, 2.0e-3, -1]).basis
    @test norm(R' * R - I) < 1.0f-6  # no cancellation near -ẑ
    # Equal field components sample one third of each principal variance, for either polarity.
    lab = Oriented(BiMaxwellian(; vth_perp = Float16(1), vth_para = Float16(4)))
    B = Float16[1e-5, 1e-5, -1e-5]
    for field in (B, -B)
        @test all(isapprox.(vth_perp_para(lab, field), sqrt(6); rtol = 0.005))
    end
    o = Oriented(d; u)
    @test pdf(o, [0.3, -1.1, 0.7] + u) ≈ pdf(d, [0.3, -1.1, 0.7])
    @test mean(Oriented(d)) == mean(d)
end

@testset "B0 is a magnitude or a vector" begin
    p = Proton
    s = Population(p, Cold(); n = 1.0e6)
    for B0 in (-5.0e-9, [0.0, 3.0e-9, 4.0e-9], [0.0, 3.0, 4.0]u"nT")
        @test alfven_speed(Plasma(s; B0)) ≈ alfven_speed(Plasma(s; B0 = 5.0e-9))
        @test gyrofrequency(p, B0) ≈ gyrofrequency(p, 5.0e-9)
    end
end

@testset "negative density" begin
    p = Proton
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
    p32 = Particle{Float32}(Proton)
    @test (@inferred BiMaxwellian(p32; T = 10.0f0)) isa BiMaxwellian{Float32}
    @test thermal_speed(p32, 1.0f4u"K") isa Float32
    @test (@inferred pdf(MaxwellJuttner(; theta = 0.1f0), 1.0f0, 1.0f0)) isa Float32
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
    p = Proton
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
    @test BiMaxwellian(p; T = Tpara, T_perp = Tperp) == BiMaxwellian(; vth_perp = v_perp, vth_para = v_para)
end

@testset "Temperatures, β, charge and current of a plasma" begin
    ions = Population(Proton, BiMaxwellian; T_perp = 100, T_para = 20, n = 5.0e6)
    plasma = Plasma(ions, Population(Electron, BiMaxwellian; T = 50, vd_para = 1.0e5, n = 5.0e6); B0 = 5.0e-9)
    @test all(temperature_perp_para(ions) .≈ (100, 20))
    @test all(temperature_perp_para(Population(Electron, MaxwellJuttner; T = 511.0e3, n = 1.0)) .≈ 511.0e3)
    @test all(temperature_perp_para(Population(Electron, Oriented(MaxwellJuttner(Electron; T = 511.0e3), [1, 2, 3]); n = 1.0), [3, 1, 0]) .≈ 511.0e3)
    @test all(plasma_beta(Plasma(plasma.populations...; B0 = [5, 0, 0]u"nT")) .≈ plasma_beta(plasma))  # gyrotropic: aligned with any B
    tilted = Population(Proton, Oriented(ions.vdf, [1, 0, 0]); ions.n)
    @test_throws ArgumentError temperature_perp_para(tilted)  # axis need not be the field
    @test all(temperature_perp_para(tilted, 5u"nT") .≈ (60, 100))
    mixed = Plasma(ions, tilted; B0 = 5u"nT")
    @test all(plasma_beta(mixed) .≈ 2MU0_SI * ions.n * E_SI .* (160, 120) ./ 25.0e-18)
    @test_throws DomainError temperature_perp_para(ions, 0)  # even where the axis is unused
    # rotated in the model frame, so a small variance survives
    @test all(vth_perp_para(Oriented(BiMaxwellian(; vth_perp = 1f0, vth_para = 1f-4), (1, 2, 3)), (1, 2, 3)) .≈ (1f0, 1f-4))
    @test all(plasma_beta(ions, plasma.B0) .≈ 2MU0_SI * 5.0e6 * E_SI .* (100, 20) ./ 25.0e-18)
    @test all(plasma_beta(plasma) .≈ 2MU0_SI * 5.0e6 * E_SI .* (150, 70) ./ 25.0e-18)
    @test charge_density(plasma) == 0
    @test current_density(plasma) ≈ [0, 0, -5.0e6 * E_SI * 1.0e5]
    @test current_density(Plasma(plasma.populations...; B0 = [-5, 0, 0]u"nT")) ≈ [5.0e6 * E_SI * 1.0e5, 0, 0]
    @test current_density(Plasma(plasma.populations...; B0 = (0, 0, 0))) == current_density(plasma)  # no direction: own ẑ
end

@testset "agyrotropy" begin
    # rotated field-aligned P⊥ = 1, P∥ = 2, P₁₂ = a: √Q = a / √(P⊥² + 2P⊥P∥)
    R = [1 0 0; 0 cosd(50) -sind(50); 0 sind(50) cosd(50)]
    P = [1 0 0; 0 2 0; 0 0 3]
    @test agyrotropy(P, 5u"nT") ≈ agyrotropy(P, (0, 0, 5))
    @test agyrotropy(Float32.(P), 5) isa Float32
    for a in (0, 0.3)
        @test agyrotropy(R * [1 a 0; a 1 0; 0 0 2] * R', R * [0, 0, 3]) ≈ a / sqrt(5) atol = 1.0e-7
    end
end

@testset "GyrotropicGridVDF tabulates a gyrotropic distribution" begin
    m = BiMaxwellian(; vth_perp = 1.3, vth_para = 0.7, vd_para = 0.4)
    v_perp, v_para = range(0, 8, 321), range(-5, 5, 401)
    F = [3.0 * pdf(m, v, u) for v in v_perp, u in v_para]
    g = GyrotropicGridVDF(v_perp, v_para, F)
    @test g.n ≈ 3 rtol = 1e-3
    gm = GyrotropicGridVDF(v_perp, collect(v_para), F)
    @test gm.v_perp isa AbstractRange && gm.v_para isa Vector && pdf(gm, 1.0, 0.5) ≈ pdf(g, 1.0, 0.5)
    gd = GyrotropicGridVDF(JLArray(collect(v_perp)), JLArray(collect(v_para)), JLArray(F))  # JLArray mimics a GPU
    vs, us = 9 * rand(20), 12 * rand(20) .- 6  # some outside the grid
    @test gd.n ≈ g.n && mean(gd) ≈ mean(g) && Array(pdf.(gd, JLArray(vs), JLArray(us))) ≈ pdf.(g, vs, us)
    g32 = GyrotropicGridVDF(Float32.(v_perp), Float32.(v_para), Float32.(F))  # e.g. Metal has no Float64
    @test g32.n isa Float32 && pdf(g32, 1.0f0, 0.5f0) isa Float32
    coldbeam = GyrotropicGridVDF(Float32[0, 1], Float32[1e7, 1e7 + 1], ones(Float32, 2, 2))
    @test last(vth_perp_para(coldbeam)) ≈ sqrt(1f0 / 6)
    @test mean(g) ≈ mean(m) atol = 1e-4
    @test all(isapprox.(vth_perp_para(g), vth_perp_para(m); rtol = 1e-3))
    @test pdf(g, v_perp[5], v_para[7]) ≈ F[5, 7] / g.n
    @test pdf(g, -(v_perp[5] + v_perp[6]) / 2, v_para[7]) ≈ (F[5, 7] + F[6, 7]) / 2g.n
    @test pdf(g, 1.0, 6.0) == 0
    # density of the interpolant: exact for f linear in v⊥ and v∥
    lin = GyrotropicGridVDF([0.0, 2.0], [-1.0, 1.0], [1.0 1.0; 3.0 3.0])
    @test lin.n ≈ 2π * 2 * (2 + 8 / 3)
    @test GyrotropicGridVDF(v_perp .* u"m/s", v_para .* u"m/s", F .* u"s^3/cm^6").n ≈ 3e12 rtol = 1e-3
    @test_throws DimensionMismatch GyrotropicGridVDF(v_perp, v_para, F')
    @test_throws ArgumentError GyrotropicGridVDF(reverse(v_perp), v_para, F)
    @test_throws DomainError GyrotropicGridVDF(v_perp .- 1, v_para, F)
    @test PlasmaPopulations.GridVDF === GyrotropicGridVDF
end

@testset "CartesianGridVDF tabulates a 3-D distribution" begin
    o = Oriented(BiMaxwellian(; vth_perp = 1.3, vth_para = 0.7), [1.0, 0.0, 1.0]; u = [0.4, -0.2, 0.1])
    vx, vy, vz = range(-6, 6, 97), range(-6, 6, 89), range(-6, 6, 81)
    F = [3.0 * pdf(o, [a, b, c]) for a in vx, b in vy, c in vz]
    g = CartesianGridVDF(vx, vy, vz, F)
    @test g.n ≈ 3 rtol = 1e-3
    @test mean(g) ≈ mean(o) atol = 1e-3
    @test cov(g) ≈ cov(o) rtol = 1e-2
    @test all(isapprox.(vth_perp_para(g, [2, 0, 2]), (1.3, 0.7); rtol = 1e-2))  # g's axes are the lab's
    @test_throws ArgumentError vth_perp_para(g)  # lab ẑ need not be the field
    @test pdf(g, [0.0, 0.0, 7.0]) == 0
    # trilinear interpolant, and its integral, exact for multilinear f on a nonuniform grid
    fl(v) = 1 + v[1] + 2v[2] + 3v[3] + v[1] * v[2] * v[3]
    xs, ys, zs = [0.0, 0.5, 2.0], [0.0, 1.0], [0.0, 0.3, 1.0]
    lin = CartesianGridVDF(xs, ys, zs, [fl((a, b, c)) for a in xs, b in ys, c in zs])
    @test lin.n ≈ 9.5
    @test pdf(lin)([1.3, 0.4, 0.7]) * lin.n ≈ fl((1.3, 0.4, 0.7))
    # and its moments: ∫ vx^a vy^b vz^c fl over [0, 2] × [0, 1] × [0, 1]
    terms = ((1, (0, 0, 0)), (1, (1, 0, 0)), (2, (0, 1, 0)), (3, (0, 0, 1)), (1, (1, 1, 1)))
    box(e) = sum(c * prod(L^(p + q + 1) / (p + q + 1) for (L, p, q) in zip((2, 1, 1), e, t)) for (c, t) in terms)
    unit(i) = ntuple(==(i), 3)
    μ = [box(unit(i)) / lin.n for i in 1:3]
    @test mean(lin) ≈ μ
    @test cov(lin) ≈ [box(unit(i) .+ unit(j)) / lin.n - μ[i] * μ[j] for i in 1:3, j in 1:3]
    gd = CartesianGridVDF(JLArray(collect(vx)), JLArray(collect(vy)), JLArray(collect(vz)), JLArray(F))
    @test gd.n ≈ g.n && mean(gd) ≈ mean(g)
    g32 = CartesianGridVDF(Float32.(vx), Float32.(vy), Float32.(vz), Float32.(F))
    @test g32.n isa Float32 && pdf(g32, Float32[1, 0, 0]) isa Float32 && eltype(cov(g32)) == Float32
    # A unit cube one Float32 ulp wide along x still has the variance of a unit cube.
    unit_axis = Float32[0, 1]
    cube = CartesianGridVDF(unit_axis .+ 1f7, unit_axis, unit_axis, ones(Float32, 2, 2, 2))
    @test cov(cube) ≈ Matrix{Float32}(I, 3, 3) / 12
    @test_throws DimensionMismatch CartesianGridVDF(vy, vx, vz, F)
    @test_throws ArgumentError CartesianGridVDF(vx, vy, vz, zero(F))
    @test_throws ArgumentError CartesianGridVDF(vx, vy, vz, -F)
    @test_throws ArgumentError CartesianGridVDF(unit_axis, unit_axis, unit_axis, fill(Inf, 2, 2, 2))
end

@testset "Grid VDF samples follow the interpolant" begin
    rng = Xoshiro(1)
    # coarse nonuniform grids, so that the interpolant differs from any smooth f
    xs, ys, zs = [-1.0, 0.5, 2.0], [0.0, 1.0], [-0.5, 0.3, 1.0]
    c = CartesianGridVDF(xs, ys, zs, [exp(a * (2b - 1)) * (c > 0 ? 4 : 1) for a in xs, b in ys, c in zs])
    vs = rand(rng, c, 400_000)
    @test all(v -> xs[1] ≤ v[1] ≤ xs[end] && ys[1] ≤ v[2] ≤ ys[end] && zs[1] ≤ v[3] ≤ zs[end], vs)
    @test mean(vs) ≈ mean(c) atol = 5e-3
    @test cov(reduce(hcat, vs); dims = 2) ≈ cov(c) rtol = 1e-2
    g = GyrotropicGridVDF([0.0, 0.5, 2.0], [-1.0, 1.0], [3.0 1.0; 0.5 2.0; 2.0 0.0])
    vs = rand(rng, g, 400_000)
    # f is linear in v∥ between its only nodes ±1, so the trapezoid is the exact v∥ integral. A nested quadgk instead
    # takes minutes to infer on Julia 1.13.
    P(a, b) = quadgk(v -> 2π * v * (b + 1) * (pdf(g, v, -1) + pdf(g, v, b)) / 2, 0, a)[1]
    for (a, b) in ((0.3, 1.0), (1.0, 0.0))
        @test mean(v -> hypot(v[1], v[2]) < a && v[3] < b, vs) ≈ P(a, b) atol = 3e-3
    end
    gd = GyrotropicGridVDF(JLArray(g.v_perp), JLArray(g.v_para), JLArray(g.f))
    @test rand(Xoshiro(2), gd, 3) == rand(Xoshiro(2), g, 3)
    @test eltype(rand(CartesianGridVDF(Float32.(xs), Float32.(ys), Float32.(zs), Float32.(c.f)))) === Float32
    uniform(f) = CartesianGridVDF(0:2.0, 0:2.0, 0:2.0, fill(f, 3, 3, 3))
    @test rand(Xoshiro(2), uniform(nextfloat(0.0)), 16) == rand(Xoshiro(2), uniform(1.0), 16)
    @test_throws DomainError rand(GyrotropicGridVDF([0.0, 1.0], [0.0, 1.0], [1.0 -1.0; 1.0 1.0]))
end

@testset "MaxwellJuttner" begin
    c = PlasmaPopulations.C_SI
    radial(d, g) = quadgk(u -> 4π * u^2 * g(u) * pdf(d, u, 0.0), 0, Inf; rtol = 1e-10)[1]
    for θ in (1e-3, 0.5, 5.0)
        d = MaxwellJuttner(; theta = θ)
        @test radial(d, one) ≈ 1 rtol = 1e-8
        @test radial(d, abs2) ≈ sum(abs2, vth_perp_para(d)) * 3 / 4 rtol = 1e-8   # ⟨u²⟩ = 3⟨u_z²⟩
        us = rand(Xoshiro(1), d, 200_000)
        @test mean(us) ≈ mean(d) atol = 5 * vth_perp_para(d)[1] / sqrt(length(us))
        for q in (0.5, 1.0, 2.0)
            P = quadgk(u -> 4π * u^2 * pdf(d, u, 0.0), 0, q * vth_perp_para(d)[1])[1]
            @test mean(u -> sqrt(sum(abs2, u)) < q * vth_perp_para(d)[1], us) ≈ P atol = 0.005
        end
        @test mean(u -> abs(u[3]) < sqrt(sum(abs2, u)) / 2, us) ≈ 1 / 2 atol = 0.005   # isotropic
    end
    @test eltype(rand(MaxwellJuttner(; theta = 0.1f0))) === Float32
    nr = MaxwellJuttner(; theta = 1e-6)
    @test pdf(nr, 0.3c * 1e-3, 0.2c * 1e-3) ≈ pdf(BiMaxwellian(; vth = c * sqrt(2e-6)), 0.3c * 1e-3, 0.2c * 1e-3) rtol = 1e-4
    @test MaxwellJuttner(Electron; T = 511u"keV").theta ≈ 1 rtol = 1e-3
    cold = MaxwellJuttner(; theta = 1e-12)
    v = (0.3c * 1e-6, 0.2c * 1e-6)
    @test pdf(cold, v...) ≈ pdf(BiMaxwellian(; vth = c * sqrt(2e-12)), v...) rtol = 1e-10
    @test_throws DomainError MaxwellJuttner(; theta = 0)
end
