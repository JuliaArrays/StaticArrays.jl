module SortTests

using StaticArrays, Test
using StaticArrays.Sort: _inttype
using Base.Order: Forward, Reverse

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

    @testset "NaNs" begin
        # Return an SVector with floats and NaNs that have random sign and payload bits.
        function floats_randnans(::Type{SVector{N, T}}, p) where {N, T}
            float_or(x, y) = reinterpret(T, |(reinterpret.(_inttype(T), (x, y))...))
            @inline function _rand(_)
                r = rand(T)
                # The bitwise or of any T with T(Inf) is either ±T(Inf) or a NaN.
                ifelse(rand(Float32) < p, float_or(typemax(T), r - T(0.5)), r)
            end
            return SVector(ntuple(_rand, Val(N)))
        end

        # Sort floats and arbitrary NaNs.
        for T in (Float16, Float32, Float64)
            buffer = Vector{T}(undef, 16)
            @test all(floats_randnans(SVector{16, T}, 0.5) for _ in 1:10_000) do a
                copyto!(buffer, a)
                isequal(sort(a), sort!(buffer))
            end
        end

        # Sort signed Infs, signed zeros, and signed NaNs with extremal payloads.
        for T in (Float16, Float32, Float64)
            U = _inttype(T)
            small_nan = reinterpret(T, reinterpret(U, typemax(T)) + one(U))
            large_nan = reinterpret(T, typemax(U))
            nans = (small_nan, large_nan, T(NaN), -small_nan, -large_nan, -T(NaN))
            (a, b, c, d) = (-T(Inf), -zero(T), zero(T), T(Inf))
            sorted = [a, b, c, d, nans..., nans...]
            @test isequal(sorted, sort(SA[nans..., d, c, b, a, nans...]))
            @test isequal(sorted, sort(SA[d, c, nans..., nans..., b, a]))
        end
    end

    # These tests are selected and modified from Julia's test/ordering.jl and test/sorting.jl.
    @testset "Base tests" begin
        # This testset partially fails on Julia versions < 1.5 because order could be
        # discarded: https://github.com/JuliaLang/julia/pull/34719
        if VERSION >= v"1.5"
            @testset "ordering" begin
                for T in (Int, Float64)
                    for (s1, rev) in enumerate([nothing, true, false])
                        for (s2, lt) in enumerate([>, <, (a, b) -> a - b > 0, (a, b) -> a - b < 0])
                            for (s3, by) in enumerate([-, +])
                                for (s4, order) in enumerate([Reverse, Forward])
                                    if isodd(s1 + s2 + s3 + s4)
                                        target = T.(SA[1, 2, 3])
                                    else
                                        target = T.(SA[3, 2, 1])
                                    end
                                    @test target == sort(T.(SA[2, 3, 1]), rev=rev, lt=lt, by=by, order=order)
                                end
                            end
                        end
                    end
                end

                @test SA[1 => 3, 2 => 5, 3 => 1] ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1]) ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1], by=first) ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1], rev=true, order=Reverse) ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1], lt= >, order=Reverse)

                @test SA[3 => 1, 1 => 3, 2 => 5] ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1], by=last) ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1], by=last, rev=true, order=Reverse) ==
                            sort(SA[1 => 3, 2 => 5, 3 => 1], by=last, lt= >, order=Reverse)
            end
        end

        @testset "sort" begin
            for T in (Int, Float64)
                @test sort(T.(SA[2,3,1])) == T.(SA[1,2,3]) == sort(T.(SA[2,3,1]); order=Forward)
                @test sort(T.(SA[2,3,1]), rev=true) == T.(SA[3,2,1]) == sort(T.(SA[2,3,1]), order=Reverse)
            end
            @test sort(SA['z':-1:'a'...]) == SA['a':'z'...]
            @test sort(SA['a':'z'...], rev=true) == SA['z':-1:'a'...]
        end

        @test sortperm(SA[2,3,1]) == SA[3,1,2]
    end

end # @testset "sort"

end # module SortTests
