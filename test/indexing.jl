@testset "Indexing" begin
    @testset "Linear getindex() on SVector" begin
        sv = SVector{4}(4,5,6,7)

        # SVector
        @test (@inferred getindex(sv, SVector(4,3,2,1))) === SVector((7,6,5,4))

        # Colon
        @test (@inferred getindex(sv,:)) === sv
    end

    @testset "Linear getindex() on SMatrix" begin
        sv = SVector{4}(4,5,6,7)
        sm = SMatrix{2,2}(4,5,6,7)

        # SVector
        @test (@inferred getindex(sm, SVector(4,3,2,1))) === SVector((7,6,5,4))

        # Colon
        @test (@inferred getindex(sm,:)) === sv
    end

    @testset "Linear getindex()/setindex!() on MVector" begin
        vec = @SVector [4,5,6,7]

        # SVector
        mv = MVector{4,Int}()
        @test (mv[SVector(1,2,3,4)] = vec; (@inferred getindex(mv, SVector(4,3,2,1)))::SVector{4,Int} == SVector((7,6,5,4)))

        # Colon
        mv = MVector{4,Int}()
        @test (mv[:] = vec; (@inferred getindex(mv, :))::SVector{4,Int} == SVector((4,5,6,7)))
    end

    @testset "Linear getindex()/setindex!() on MMatrix" begin
        vec = @SVector [4,5,6,7]

        # SVector
        mm = MMatrix{2,2,Int}()
        @test (mm[SVector(1,2,3,4)] = vec; (@inferred getindex(mm, SVector(4,3,2,1)))::SVector{4,Int} == SVector((7,6,5,4)))

        # Colon
        mm = MMatrix{2,2,Int}()
        @test (mm[:] = vec; (@inferred getindex(mm, :))::SVector{4,Int} == SVector((4,5,6,7)))
    end

    @testset "Linear getindex()/setindex!() with a SVector on an Array" begin
        v = [11,12,13]

        @test (v[SVector(2,3)] = [22,23]; (v[2] == 22) & (v[3] == 23))
    end

    @testset "2D getindex() on SMatrix" begin
        sm = @SMatrix [1 3; 2 4]

        # Scalar
        @test sm[1,1] === 1
        @test sm[2,1] === 2
        @test sm[1,2] === 3
        @test sm[2,2] === 4

        # SVector, scalar
        @test (@inferred getindex(sm, SVector(2,1), SVector(2,1))) === @SMatrix [4 2; 3 1]
        @test (@inferred getindex(sm, 1, SVector(1,2))) === @SVector [1,3]
        @test (@inferred getindex(sm, SVector(1,2), 1)) === @SVector [1,2]

        # Colon
        @test (@inferred getindex(sm, :, :)) === @SMatrix [1 3; 2 4]
        @test (@inferred getindex(sm, SVector(2,1), :)) === @SMatrix [2 4; 1 3]
        @test (@inferred getindex(sm, :, SVector(2,1))) === @SMatrix [3 1; 4 2]
        @test (@inferred getindex(sm, 1, :)) === @SVector [1,3]
        @test (@inferred getindex(sm, :, 1)) === @SVector [1,2]
    end

    @testset "2D getindex()/setindex! on MMatrix" begin
        sm = @MMatrix [1 3; 2 4]

        # Tuple, scalar
        @test (mm = MMatrix{2,2,Int}(); mm[SVector(2,1),SVector(2,1)] = sm[SVector(2,1),SVector(2,1)]; (@inferred getindex(mm, SVector(2,1), SVector(2,1)))::SMatrix == @SMatrix [4 2; 3 1])
        @test (mm = MMatrix{2,2,Int}(); mm[1,SVector(1,2)] = sm[1,SVector(1,2)]; (@inferred getindex(mm, 1, SVector(1,2)))::SVector == @SVector [1,3])
        @test (mm = MMatrix{2,2,Int}(); mm[SVector(1,2),1] = sm[SVector(1,2),1]; (@inferred getindex(mm, SVector(1,2), 1))::SVector == @SVector [1,2])

        # Colon
        @test (mm = MMatrix{2,2,Int}(); mm[:,:] = sm[:,:]; (@inferred getindex(mm, :, :))::SMatrix == @MMatrix [1 3; 2 4])
        @test (mm = MMatrix{2,2,Int}(); mm[SVector(2,1),:] = sm[SVector(2,1),:]; (@inferred getindex(mm, SVector(2,1), :))::SMatrix == @SMatrix [2 4; 1 3])
        @test (mm = MMatrix{2,2,Int}(); mm[:,SVector(2,1)] = sm[:,SVector(2,1)]; (@inferred getindex(mm, :, SVector(2,1)))::SMatrix == @SMatrix [3 1; 4 2])
        @test (mm = MMatrix{2,2,Int}(); mm[1,:] = sm[1,:]; (@inferred getindex(mm, 1, :))::SVector == @SVector [1,3])
        @test (mm = MMatrix{2,2,Int}(); mm[:,1] = sm[:,1]; (@inferred getindex(mm, :, 1))::SVector == @SVector [1,2])
    end

    @testset "3D scalar indexing" begin
        sa = SArray{Tuple{2,2,2}, Int}([i*j*k for i = 1:2, j = 2:3, k=3:4])

        @test sa[1,1,2] === 8
        @test sa[1,2,1] === 9
        @test sa[2,1,1] === 12

        ma = MArray{Tuple{2,2,2}, Int}()
        @test (ma[1,1,2] = 8; ma[1,1,2] === 8)
        @test (ma[1,2,1] = 9; ma[1,2,1] === 9)
        @test (ma[2,1,1] = 12; ma[2,1,1] === 12)
    end

    @testset "4D scalar indexing" begin
        sa = SArray{Tuple{2,2,2,2}, Int}([i*j*k*l for i = 1:2, j = 2:3, k=3:4, l=4:5])

        @test sa[1,1,1,2] === 30
        @test sa[1,1,2,1] === 32
        @test sa[1,2,1,1] === 36
        @test sa[2,1,1,1] === 48

        ma = MArray{Tuple{2,2,2,2}, Int}()
        @test (ma[1,1,1,2] = 30; ma[1,1,1,2] === 30)
        @test (ma[1,1,2,1] = 32; ma[1,1,2,1] === 32)
        @test (ma[1,2,1,1] = 36; ma[1,2,1,1] === 36)
        @test (ma[2,1,1,1] = 48; ma[2,1,1,1] === 48)
    end

    @testset "Indexing with empty vectors" begin
        a = randn(2,2)
        @test a[SVector{0,Int}()] == SVector{0,Float64}(())
        @test a[SVector{0,Int}(),SVector{0,Int}()] == SMatrix{0,0,Float64,0}(())
        b = copy(a)
        a[SVector{0,Int}()] = 5.0
        @test b == a
    end

    @testset "inferabilty of index_sizes helper" begin
        # see JuliaLang/julia#21244
        # it's not about inferring the correct type, but about inference throwing an error
        @test code_warntype(DevNull, StaticArrays.index_sizes, Tuple{Vararg{Any}}) == nothing
    end
end
