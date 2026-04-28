# Run from the repo root with:
#   julia --project=test/ext/mooncake test/ext/mooncake/mooncake.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using AllocCheck, JET, Mooncake, StableRNGs, StaticArrays, Test
using Mooncake.TestUtils: test_rule, test_tangent_interface, test_tangent_splitting

# The extension covers `SArray` only by design (it's the by-value, leaf-tangent
# case); `MArray` falls through to Mooncake's generic mutable-array handling.
@testset verbose=true "Mooncake integration" begin
    cases = Any[
        SVector{3,Float64}(1.0, 2.0, 3.0),
        SVector{2,Float32}(1.0f0, -2.0f0),
        SMatrix{2,2,Float64}(1.0, 2.0, 3.0, 4.0),
        SVector{2,ComplexF64}(1.0 + 2.0im, -3.0 + 1.0im),
        SVector{1,ComplexF32}(0.5f0 + 0.25f0im),
    ]

    @testset "tangent interface for $(typeof(p))" for p in cases
        rng = StableRNG(123)
        test_tangent_interface(rng, p)
        test_tangent_splitting(rng, p)
    end

    @testset "rrule!! getindex $(typeof(p))" for p in cases
        for i in eachindex(p)
            test_rule(StableRNG(123), getindex, p, i; is_primitive=true)
        end
    end

    @testset "rrule!! _new_ construction" begin
        # `_new_` is the primitive that IR normalisation lowers `SArray(...)`
        # construction calls into; test the SArray-specific method directly.
        new_cases = Any[
            (SVector{3,Float64}, (1.0, 2.0, 3.0)),
            (SMatrix{2,2,Float64,4}, (1.0, 2.0, 3.0, 4.0)),
            (SVector{2,ComplexF64}, (1.0 + 2.0im, -3.0 + 1.0im)),
        ]
        for (P, data) in new_cases
            test_rule(StableRNG(123), Mooncake._new_, P, data; is_primitive=true)
        end
    end

    @testset "end-to-end gradient" begin
        f(x) = x[1]^2 + 2 * x[2] * x[3]
        x = SVector{3,Float64}(1.5, -2.0, 0.5)
        cache = Mooncake.prepare_gradient_cache(f, x)
        val, (_, dx) = Mooncake.value_and_gradient!!(cache, f, x)
        @test val ≈ f(x)
        @test dx ≈ SVector{3,Float64}(2 * x[1], 2 * x[3], 2 * x[2])
    end
end
