@testset "Map, reduce, mapreduce, broadcast" begin
    @testset "map and map!" begin
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv = MVector{4, Int}()

        normal_v1 = [2,4,6,8]
        normal_v2 = [4,3,2,1]

        @test @inferred(map(-, v1)) === @SVector [-2, -4, -6, -8]
        @test @inferred(map(+, v1, v2)) === @SVector [6, 7, 8, 9]
        @test @inferred(map(+, normal_v1, v2)) === @SVector [6, 7, 8, 9]
        @test @inferred(map(+, v1, normal_v2)) === @SVector [6, 7, 8, 9]

        map!(+, mv, v1, v2)
        @test mv == @MVector [6, 7, 8, 9]
        mv2 = MVector{4, Int}()
        map!(x->x^2, mv2, v1)
        @test mv2 == @MVector [4, 16, 36, 64]
        mv3 = MVector{4, Int}()
        v3 = @SVector [1, 2, 3, 4]
        map!(+, mv3, v1, v2, v3)
        @test mv3 == @MVector [7, 9, 11, 13]
    end

    @testset "[map]reduce and [map]reducedim" begin
        a = rand(4,3); sa = SMatrix{4,3}(a)
        v1 = [2,4,6,8]; sv1 = SVector{4}(v1)
        v2 = [4,3,2,1]; sv2 = SVector{4}(v2)
        @test reduce(+, sv1) === reduce(+, v1)
        @test reduce(+, 0, sv1) === reduce(+, 0, v1)
        @test reducedim(max, sa, Val{1}, -1.) == reducedim(max, a, 1, -1.)
        @test reducedim(max, sa, Val{2}, -1.) == reducedim(max, a, 2, -1.)
        @test mapreduce(-, +, sv1) === mapreduce(-, +, v1)
        @test mapreduce(-, +, 0, sv1) === mapreduce(-, +, 0, v1)
        @test mapreduce(*, +, sv1, sv2) === 40
        @test mapreduce(*, +, 0, sv1, sv2) === 40
        @test mapreducedim(x->x^2, max, sa, Val{1}, -1.) == mapreducedim(x->x^2, max, a, 1, -1.)
        @test mapreducedim(x->x^2, max, sa, Val{2}, -1.) == mapreducedim(x->x^2, max, a, 2, -1.)
    end

    @testset "implemented by [map]reduce and [map]reducedim" begin
        a = randn(2,2,2); sa = SArray{Tuple{2,2,2}}(a)
        b = rand(Bool,2,2,2); sb = SArray{Tuple{2,2,2}}(b)
        z = zeros(2,2,2); sz = SArray{Tuple{2,2,2}}(z)

        @test iszero(sz) == iszero(z)

        @test sum(sa) === sum(a)
        @test sum(abs2, sa) === sum(abs2, a)
        @test sum(sa, Val{2}) == sum(a, 2)
        @test sum(abs2, sa, Val{2}) == sum(abs2, a, 2)

        @test prod(sa) === prod(a)
        @test prod(abs2, sa) === prod(abs2, a)
        @test prod(sa, Val{2}) == prod(a, 2)
        @test prod(abs2, sa, Val{2}) == prod(abs2, a, 2)

        @test count(sb) === count(b)
        @test count(x->x>0, sa) === count(x->x>0, a)
        # @test count(sb, Val{2}) == count(b, 2)
        # @test count(x->x>0, sa, Val{2}) == count(x->x>0, a, 2)

        @test all(sb) === all(b)
        @test all(x->x>0, sa) === all(x->x>0, a)
        @test all(sb, Val{2}) == all(b, 2)
        @test all(x->x>0, sa, Val{2}) == all(x->x>0, a, 2)

        @test any(sb) === any(b)
        @test any(x->x>0, sa) === any(x->x>0, a)
        @test any(sb, Val{2}) == any(b, 2)
        @test any(x->x>0, sa, Val{2}) == any(x->x>0, a, 2)

        @test mean(sa) === mean(a)
        @test mean(abs2, sa) === mean(abs2, a)
        @test mean(sa, Val{2}) == mean(a, 2)
        # @test mean(abs2, sa, Val{2}) == mean(abs2, a, 2)

        @test minimum(sa) === minimum(a)
        @test minimum(abs2, sa) === minimum(abs2, a)
        @test minimum(sa, Val{2}) == minimum(a, 2)
        @test minimum(abs2, sa, Val{2}) == minimum(abs2, a, 2)

        @test maximum(sa) === maximum(a)
        @test maximum(abs2, sa) === maximum(abs2, a)
        @test maximum(sa, Val{2}) == maximum(a, 2)
        @test maximum(abs2, sa, Val{2}) == maximum(abs2, a, 2)

        @test diff(sa, Val{1}) == a[2:end,:,:] - a[1:end-1,:,:]
        @test diff(sa, Val{2}) == a[:,2:end,:] - a[:,1:end-1,:]
        @test diff(sa, Val{3}) == a[:,:,2:end] - a[:,:,1:end-1]

        # as of Julia v0.6, diff() for regular Array is defined only for vectors and matrices
        m = randn(4,3); sm = SMatrix{4,3}(m)
        @test diff(sm, Val{1}) == diff(m, 1) == diff(sm) == diff(m)
        @test diff(sm, Val{2}) == diff(m, 2)
    end

    @testset "broadcast and broadcast!" begin
        c = 2
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]
        mv = MVector{4, Int}()
        mm = MMatrix{4, 2, Int}()
        M = @SMatrix [1 2; 3 4; 5 6; 7 8]

        @test_throws DimensionMismatch broadcast(+, v1, @SMatrix [4 3 2 1; 5 6 7 8])

        @test @inferred(broadcast(convert, Float64, v1)) == convert(Vector{Float64}, [2,4,6,8])

        @test @inferred(broadcast(-, v1)) === map(-, v1)

        @test @inferred(broadcast(+, v1, c)) === @SVector [4, 6, 8, 10]
        @test @inferred(broadcast(+, v1, v2)) === map(+, v1, v2)
        @test @inferred(broadcast(+, v1, M)) === @SMatrix [3 4; 7 8; 11 12; 15 16]

        @test_throws DimensionMismatch broadcast!(-, MVector{5, Int}(), v1)

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
end
