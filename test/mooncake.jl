using Pkg
Pkg.add("Mooncake")

using StaticArrays
using Mooncake
using Test

@testset verbose=true "Mooncake integration" begin
    f(x) = sum(abs2, x)
    config = Mooncake.Config(; friendly_tangents=true)
    @testset "$(typeof(x))" for x in [
        SVector(1.0, 2.0),
        MVector(1.0, 2.0) ,
        SMatrix{2,2}(1.0, 2.0, 3.0, 4.0),
        MMatrix{2,2}(1.0, 2.0, 3.0, 4.0)
    ]
        cache = prepare_gradient_cache(f, zero(x); config)
        val, grads = value_and_gradient!!(cache, f, x)
        g = grads[2]
        @test g isa typeof(x)
        @test g == 2x
    end
end
