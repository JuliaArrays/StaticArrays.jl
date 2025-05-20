using StaticArrays, Test
using Statistics: mean

@testset "Map, reduce, mapreduce, broadcast" begin
    @testset "map and map!" begin
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv1 = @MVector [2,4,6,8]
        mv2 = @MVector [4,3,2,1]
        mv = MVector{4, Int}(undef)

        normal_v1 = [2,4,6,8]
        normal_v2 = [4,3,2,1]

        @test @inferred(map(-, v1)) === @SVector [-2, -4, -6, -8]
        @test @inferred(map(+, v1, v2)) === @SVector [6, 7, 8, 9]
        @test @inferred(map(+, normal_v1, v2)) === @SVector [6, 7, 8, 9]
        @test @inferred(map(+, v1, normal_v2)) === @SVector [6, 7, 8, 9]

        # Make sure similar_type is based on first <: StaticArray
        @test @inferred(map(+, normal_v1, mv2))::MVector{4,Int} == @MVector [6, 7, 8, 9]
        @test @inferred(map(+, mv1, normal_v2))::MVector{4,Int} == @MVector [6, 7, 8, 9]

        @test map!(+, mv, v1, v2) === mv
        @test mv == @MVector [6, 7, 8, 9]
        mv2 = MVector{4, Int}(undef)
        map!(x->x^2, mv2, v1)
        @test mv2 == @MVector [4, 16, 36, 64]
        mv3 = MVector{4, Int}(undef)
        v3 = @SVector [1, 2, 3, 4]
        map!(+, mv3, v1, v2, v3)
        @test mv3 == @MVector [7, 9, 11, 13]

        if VERSION >= v"1.12.0-beta3"
            @testset "map!(function, array)" begin
                local mv = MVector(1,2,3)
                map!(x->x^2, mv)
                @test mv == SA[1,4,9]
            end
        end

        # Output eltype for empty cases #528
        @test @inferred(map(/, SVector{0,Int}(), SVector{0,Int}())) === SVector{0,Float64}()
        @test @inferred(map(+, SVector{0,Int}(), SVector{0,Float32}())) === SVector{0,Float32}()
        @test @inferred(map(length, SVector{0,String}())) === SVector{0,Int}()
        # similar_type based on first <: StaticArray
        @test @inferred(map(+, MVector{0,Int}(), Int[]))::MVector{0,Int} == MVector{0,Int}()
        @test @inferred(map(+, Int[], MVector{0,Int}()))::MVector{0,Int} == MVector{0,Int}()
    end

    @testset "[map]reduce and [map]reducedim" begin
        a = rand(4,3); sa = SMatrix{4,3}(a); (I,J) = size(a)
        v1 = [2,4,6,8]; sv1 = SVector{4}(v1)
        v2 = [4,3,2,1]; sv2 = SVector{4}(v2)
        @test reduce(+, sv1) === reduce(+, v1)
        @test reduce(+, sv1; init=0) === reduce(+, v1; init=0)
        @test reduce(max, sa; dims=Val(1), init=-1.) === SMatrix{1,J}(reduce(max, a, dims=1, init=-1.))
        @test reduce(max, sa; dims=1, init=-1.) === SMatrix{1,J}(reduce(max, a, dims=1, init=-1.))
        @test reduce(max, sa; dims=2, init=-1.) === SMatrix{I,1}(reduce(max, a, dims=2, init=-1.))
        @test mapreduce(-, +, sv1) === mapreduce(-, +, v1)
        @test mapreduce(-, +, sv1; init=0) === mapreduce(-, +, v1, init=0)
        @test mapreduce(*, +, sv1, sv2) === 40
        @test mapreduce(*, +, sv1, sv2; init=0) === 40
        @test mapreduce(x->x^2, max, sa; dims=Val(1), init=-1.) == SMatrix{1,J}(mapreduce(x->x^2, max, a, dims=1, init=-1.))
        @test mapreduce(x->x^2, max, sa; dims=1, init=-1.) == SMatrix{1,J}(mapreduce(x->x^2, max, a, dims=1, init=-1.))
        @test mapreduce(x->x^2, max, sa; dims=2, init=-1.) == SMatrix{I,1}(mapreduce(x->x^2, max, a, dims=2, init=-1.))
    end

    @testset "[map]foldl" begin
        a = rand(4,3)
        v1 = [2,4,6,8]; sv1 = SVector{4}(v1)
        @test foldl(+, sv1) === foldl(+, v1)
        @test foldl(+, sv1; init=0) === foldl(+, v1; init=0)
        @test mapfoldl(-, +, sv1) === mapfoldl(-, +, v1)
        @test mapfoldl(-, +, sv1; init=0) === mapfoldl(-, +, v1, init=0)
    end

    @testset "empty array" begin
        # issue #778
        @test iszero(SVector{0,Int}())

        @testset "$fold" for fold in [reduce, foldl]
            @test fold(+, SVector{0,Bool}()) === 0
            @test fold(nothing, SVector{0,Int}(), init=:INIT) === :INIT
        end

        @testset "$mapfold" for mapfold in [mapreduce, mapfoldl]
            @test mapfold(identity, +, SVector{0,Bool}()) === 0
            @test mapfold(abs, +, SVector{0,Bool}()) === 0
            @test mapfold(nothing, nothing, SVector{0,Int}(), init=:INIT) === :INIT
        end

        @test mapreduce(
            (a, b) -> a + b,
            (a, b) -> a * b,
            SVector{0,Int}(),
            SVector{0,Int}();
            init = :INIT,
        ) == :INIT

        # When there are multiple inputs, the error is thrown by
        # StaticArrays.jl:
        @test_throws(
            ArgumentError("reducing over an empty collection is not allowed"),
            mapreduce((a, b) -> a + b, (a, b) -> a * b, SVector{0,Int}(), SVector{0,Int}())
        )

        # When the mapping and/or reducing functions are unsupported,
        # the error is thrown by `Base.mapreduce_empty`:
        if Base.VERSION >= v"1.8.0-DEV.363"
            @test_throws(
                "reducing over an empty collection is not allowed",
                mapreduce(nothing, nothing, SVector{0,Int}())
            )
        else
            @test_throws(
                ArgumentError("reducing over an empty collection is not allowed"),
                mapreduce(nothing, nothing, SVector{0,Int}())
            )
        end
    end

    @testset "implemented by [map]reduce and [map]reducedim" begin
        I, J, K = 2, 2, 2
        OSArray = SArray{Tuple{I,J,K}}  # original
        RSArray1 = SArray{Tuple{1,J,K}}  # reduced in dimension 1
        RSArray2 = SArray{Tuple{I,1,K}}  # reduced in dimension 2
        RSArray3 = SArray{Tuple{I,J,1}}  # reduced in dimension 3
        a = randn(I,J,K); sa = OSArray(a)
        b = rand(Bool,I,J,K); sb = OSArray(b)
        z = zeros(I,J,K); sz = OSArray(z)

        @test iszero(sz) == iszero(z)

        @test sum(sa) === sum(a)
        @test sum(abs2, sa) === sum(abs2, a)
        @test sum(sa, dims=2) === RSArray2(sum(a, dims=2))
        @test sum(sa, dims=Val(2)) === RSArray2(sum(a, dims=2))
        @test sum(abs2, sa; dims=2) === RSArray2(sum(abs2, a, dims=2))
        @test sum(abs2, sa; dims=Val(2)) === RSArray2(sum(abs2, a, dims=2))
        @test sum(sa, init=2) == sum(a, init=2) ≈ sum(sa) + 2 # Float64 is non-associative
        @test sum(sb, init=2) == sum(b, init=2) == sum(sb) + 2

        @test prod(sa) === prod(a)
        @test prod(abs2, sa) === prod(abs2, a)
        @test prod(sa, dims=Val(2)) === RSArray2(prod(a, dims=2))
        @test prod(abs2, sa, dims=Val(2)) === RSArray2(prod(abs2, a, dims=2))
        @test prod(sa, init=2) == prod(a, init=2) ≈ 2*prod(sa) # Float64 is non-associative
        @test prod(sb, init=2) == prod(b, init=2) == 2*prod(sb)

        @test count(sb) === count(b)
        @test count(x->x>0, sa) === count(x->x>0, a)
        @test count(sb, dims=Val(2)) === RSArray2(reshape([count(b[i,:,k]) for i = 1:I, k = 1:K], (I,1,K)))
        @test count(x->x>0, sa, dims=Val(2)) === RSArray2(reshape([count(x->x>0, a[i,:,k]) for i = 1:I, k = 1:K], (I,1,K)))
        @test count(sb, init=3) == count(b, init=3) == count(sb) + 3
        @test count(x->x>0, sa, init=-2) == count(x->x>0, a, init=-2) == count(x->x>0, sa) - 2

        @test all(sb) === all(b)
        @test all(x->x>0, sa) === all(x->x>0, a)
        @test all(sb, dims=Val(2)) === RSArray2(all(b, dims=2))
        @test all(x->x>0, sa, dims=Val(2)) === RSArray2(all(x->x>0, a, dims=2))

        @test any(sb) === any(b)
        @test any(x->x>0, sa) === any(x->x>0, a)
        @test any(sb, dims=Val(2)) === RSArray2(any(b, dims=2))
        @test any(x->x>0, sa, dims=Val(2)) === RSArray2(any(x->x>0, a, dims=2))

        @test all(in(x, sa) for x in sa)
        @test all(in(x, sa) === in(x, a) for x in randn(10))

        @test mean(sa) === mean(a)
        @test mean(abs2, sa) === mean(abs2, a)
        @test mean(sa, dims=Val(2)) === RSArray2(mean(a, dims=2))
        @test mean(abs2, sa, dims=Val(2)) === RSArray2(mean(abs2.(a), dims=2))

        @test minimum(sa) === minimum(a)
        @test minimum(abs2, sa) === minimum(abs2, a)
        @test minimum(sa, dims=Val(2)) === RSArray2(minimum(a, dims=2))
        @test minimum(abs2, sa, dims=Val(2)) === RSArray2(minimum(abs2, a, dims=2))

        @test maximum(sa) === maximum(a)
        @test maximum(abs2, sa) === maximum(abs2, a)
        @test maximum(sa, dims=Val(2)) === RSArray2(maximum(a, dims=2))
        @test maximum(abs2, sa, dims=Val(2)) === RSArray2(maximum(abs2, a, dims=2))

        @test diff(sa, dims=Val(1)) === RSArray1(a[2:end,:,:] - a[1:end-1,:,:])
        @test diff(sa, dims=1) === RSArray1(a[2:end,:,:] - a[1:end-1,:,:])
        @test diff(sa, dims=Val(2)) === RSArray2(a[:,2:end,:] - a[:,1:end-1,:])
        @test diff(sa, dims=2) === RSArray2(a[:,2:end,:] - a[:,1:end-1,:])
        @test diff(sa, dims=Val(3)) === RSArray3(a[:,:,2:end] - a[:,:,1:end-1])
        @test diff(sa, dims=3) === RSArray3(a[:,:,2:end] - a[:,:,1:end-1])

        v = randn(4); sv = SVector{4}(v)
        m = randn(4,3); sm = SMatrix{4,3}(m)
        @test diff(sv, dims=Val(1)) == diff(sv, dims=1) == diff(sv) == diff(v)
        @test diff(sm, dims=Val(1)) == diff(sm, dims=1) == diff(m, dims=1)
        @test diff(sm, dims=Val(2)) == diff(sm, dims=2) == diff(m, dims=2)

        # diff on nested arrays should result in the correct eltype
        @test @inferred(diff(SA[SA[1,2],SA[3,4]])) === SVector{1,SVector{2,Int}}(((2,2),))
        @test @inferred(diff(SA[[1,2],[3,4]]))::SVector{1,Vector{Int}} == SA[[2,2]]
        # For larger cases, check eltype infers correctly
        @test @inferred(diff(zeros(SVector{100,SVector{2,Int}}))) == fill(SA[0,0],99)
    end

    @testset "broadcast and broadcast!" begin
        c = 2
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv = MVector{4, Int}(undef)
        mm = MMatrix{4, 2, Int}(undef)
        M = @SMatrix [1 2; 3 4; 5 6; 7 8]

        @test_throws DimensionMismatch broadcast(+, v1, @SMatrix [4 3 2 1; 5 6 7 8])

        @test @inferred(broadcast(convert, Float64, v1)) === SVector{4,Float64}(2,4,6,8)
        # issue 329
        @test @inferred(broadcast(round, Int, SVector(1.0,2.0))) === SVector{2,Int}(1,2)

        @test @inferred(broadcast(-, v1)) === map(-, v1)

        @test @inferred(broadcast(+, v1, c)) === @SVector [4, 6, 8, 10]
        @test @inferred(broadcast(+, v1, v2)) === map(+, v1, v2)
        @test @inferred(broadcast(+, v1, M)) === @SMatrix [3 4; 7 8; 11 12; 15 16]

        @test_throws DimensionMismatch broadcast!(-, MVector{5, Int}(undef), v1)

        broadcast!(-, mv, v1)
        @test mv == @MVector [-2, -4, -6, -8]

        broadcast!(+, mv, v1, v2)
        @test mv == @MVector [6, 7, 8, 9]

        broadcast!(+, mm, v1, M)
        @test mm == @MMatrix [3 4; 7 8; 11 12; 15 16]
        # issue #103
        @test map(+, M, M) == [2 4; 6 8; 10 12; 14 16]

        @test ((@SVector Int64[]) + (@SVector Int64[])) == (@SVector Int64[])
    end
    @testset "Nested SVectors" begin
        # issue #593
        v = SVector(SVector(3, 2), SVector(5, 7))
        @test @inferred(v + v) == SVector(SVector(6, 4), SVector(10, 14))
        v = SVector(SVector(3, 2, 1), SVector(5, 7, 9))
        @test @inferred(v + v) == SVector(SVector(6, 4, 2), SVector(10, 14, 18))
    end
    @testset "hcat and vcat" begin
        # issue #641
        v = SVector([1,2], [3,4])
        @test reduce(vcat, v) == [1,2,3,4]
        @test reduce(hcat, v) == [1 3; 2 4]
        v2 = SVector(SVector(1,2), SVector(3,4))
        @test @inferred(reduce(vcat, v2)) === @SVector [1,2,3,4]
        @test @inferred(reduce(hcat, v2)) === @SMatrix [1 3; 2 4]
    end
    @testset "map over enumerate" begin
        # issue #1106
        v = @SVector [1, -2, 3, -4]
        m = @SMatrix [1 -2; 3 -4]
        v0 = SVector{0,Float64}()
        m0 = SMatrix{0,0,Float64}()
        @test @inferred(map(f -> f[1] * f[2], enumerate_static(v))) === @SVector [1, -4, 9, -16]
        @test @inferred(map(f -> f[1] * f[2], enumerate_static(m))) === @SMatrix [1 -6; 6 -16]
        @test @inferred(map(f -> f, enumerate_static(v0))) === SVector{0,Tuple{Int,Float64}}()
        @test @inferred(map(f -> f, enumerate_static(m0))) === SMatrix{0,0,Tuple{Int,Float64}}()
    end
    @testset "reduce over empty array" begin
        # issue #1114
        @test (@inferred reduce(|,zeros(SMatrix{0,3,Bool}); dims=Val(1), init=false)) ==
            reduce(|,zeros(Bool,0,3); dims=1, init=false)
        @test reduce(|,zeros(SMatrix{0,3,Bool}); dims=1, init=false) ==
            reduce(|,zeros(Bool,0,3); dims=1, init=false)
    end
end
