using StaticArrays, Test
using Statistics: Statistics, median, median!, middle

@testset "sort" begin

    @testset "basics" for T in (Int, Float64)
        for N in (0, 1, 2, 3, 10, 20, 30)
            vs = rand(SVector{N,T})
            vm = MVector{N,T}(vs)
            vref = sort(Vector(vs))

            @test @inferred(sort(vs)) isa SVector
            @test @inferred(sort(vs, alg=QuickSort)) isa SVector
            @test @inferred(sort(vm)) isa MVector
            @test vref == sort(vs)
            @test vref == sort(vm)

            # @allocated seems broken since 1.4
            #N <= 20 && @test 0 == @allocated sort(vs)
        end
    end

    @testset "sortperm" for T in (Int, Float64)
        for N in (0, 1, 2, 3, 10, 20, 30)
            v = rand(SVector{N,T})
            vp = sortperm(v)
            @test v[vp] == sort(v)
        end

        # stability
        @test sortperm(SA[1, 1, 1, 0]) == SA[4, 1, 2, 3]
    end

    @testset "median" begin
        @test_throws ArgumentError median(SA[])
        @test ismissing(median(SA[1, missing]))
        @test isnan(median(SA[1., NaN]))

        @testset for T in (Int, Float64)
            for N in (1, 2, 3, 10, 20, 30)
                v = rand(SVector{N,T})
                mref = median(Vector(v))

                @test @inferred(median(v) == mref)
            end
        end

        # Tests based on upstream `Statistics.jl`.
        # https://github.com/JuliaStats/Statistics.jl/blob/d49c2bf4f81e1efb4980a35fe39c815ef8396297/test/runtests.jl#L31-L92
        @test median(SA[1.]) === 1.
        @test median(SA[1.,3]) === 2.
        @test median(SA[1.,3,2]) === 2.

        @test median(SA[1,3,2]) === 2.0
        @test median(SA[1,3,2,4]) === 2.5

        @test median(SA[0.0,Inf]) == Inf
        @test median(SA[0.0,-Inf]) == -Inf
        @test median(SA[0.,Inf,-Inf]) == 0.0
        @test median(SA[1.,-1.,Inf,-Inf]) == 0.0
        @test isnan(median(SA[-Inf,Inf]))

        X = SA[2 3 1 -1; 7 4 5 -4]
        @test all(median(X, dims=2) .== SA[1.5, 4.5])
        @test all(median(X, dims=1) .== SA[4.5 3.5 3.0 -2.5])
        @test X == SA[2 3 1 -1; 7 4 5 -4] # issue #17153

        @test_throws ArgumentError median(SA[])
        @test isnan(median(SA[NaN]))
        @test isnan(median(SA[0.0,NaN]))
        @test isnan(median(SA[NaN,0.0]))
        @test isnan(median(SA[NaN,0.0,1.0]))
        @test isnan(median(SA{Any}[NaN,0.0,1.0]))
        @test isequal(median(SA[NaN 0.0; 1.2 4.5], dims=2), reshape(SA[NaN; 2.85], 2, 1))

        # the specific NaN value is propagated from the input
        @test median(SA[NaN]) === NaN
        @test median(SA[0.0,NaN]) === NaN
        @test median(SA[0.0,NaN,NaN]) === NaN
        @test median(SA[-NaN]) === -NaN
        @test median(SA[0.0,-NaN]) === -NaN
        @test median(SA[0.0,-NaN,-NaN]) === -NaN

        @test ismissing(median(SA[1, missing]))
        @test ismissing(median(SA[1, 2, missing]))
        @test ismissing(median(SA[NaN, 2.0, missing]))
        @test ismissing(median(SA[NaN, missing]))
        @test ismissing(median(SA[missing, NaN]))
        @test ismissing(median(SA{Any}[missing, 2.0, 3.0, 4.0, NaN]))
        @test median(skipmissing(SA[1, missing, 2])) === 1.5

        @test median!(Base.copymutable(SA[1 2 3 4])) == 2.5
        @test median!(Base.copymutable(SA[1 2; 3 4])) == 2.5

        @test @inferred(median(SA{Float16}[1, 2, NaN])) === Float16(NaN)
        @test @inferred(median(SA{Float16}[1, 2, 3]))   === Float16(2)
        @test @inferred(median(SA{Float32}[1, 2, NaN])) === NaN32
        @test @inferred(median(SA{Float32}[1, 2, 3]))   === 2.0f0

        # custom type implementing minimal interface
        struct A
            x
        end
        Statistics.middle(x::A, y::A) = A(middle(x.x, y.x))
        Base.isless(x::A, y::A) = isless(x.x, y.x)
        @test median(SA[A(1), A(2)]) === A(1.5)
        @test median(SA{Any}[A(1), A(2)]) === A(1.5)
    end

end
