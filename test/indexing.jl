@testset "Indexing" begin
    @testset "Linear getindex() on SVector" begin
        sv = SVector{4}(4,5,6,7)

        # Tuple
        @test (@inferred getindex(sv, (4,3,2,1))) === SVector((7,6,5,4))

        # Colon
        @test (@inferred getindex(sv,:)) === sv
    end

    @testset "Linear getindex()/setindex!() on MVector" begin
        vec = @SVector [4,5,6,7]

        # Tuple
        mv = MVector{4,Int}()
        @test (mv[(1,2,3,4)] = vec; (@inferred getindex(mv, (4,3,2,1)))::MVector{4,Int} == MVector((7,6,5,4)))

        # Colon
        mv = MVector{4,Int}()
        @test (mv[:] = vec; (@inferred getindex(mv, :))::MVector{4,Int} == MVector((4,5,6,7)))
    end

    @testset "Linear getindex()/setindex!() with a tuple on an Array" begin
        v = [11,12,13]
        m = [1.0 2.0; 3.0 4.0]

        @test v[(2,3)] === (12, 13)
        @test m[(2,3)] === (3.0, 2.0)

        @test (v[(2,3)] = [22,23]; (v[2] == 22) & (v[3] == 23))

    end

    @testset "2D getindex() on SMatrix" begin
        sm = @SMatrix [1 3; 2 4]

        # Scalar
        @test sm[1,1] === 1
        @test sm[2,1] === 2
        @test sm[1,2] === 3
        @test sm[2,2] === 4

        # Tuple, scalar
        @test (@inferred getindex(sm, (2,1), (2,1))) === @SMatrix [4 2; 3 1]
        @test (@inferred getindex(sm, 1, (1,2))) === @SVector [1,3]
        @test (@inferred getindex(sm, (1,2), 1)) === @SVector [1,2]

        # Colon
        @test (@inferred getindex(sm, :, :)) === @SMatrix [1 3; 2 4]
        @test (@inferred getindex(sm, (2,1), :)) === @SMatrix [2 4; 1 3]
        @test (@inferred getindex(sm, :, (2,1))) === @SMatrix [3 1; 4 2]
        @test (@inferred getindex(sm, 1, :)) === @SVector [1,3]
        @test (@inferred getindex(sm, :, 1)) === @SVector [1,2]
    end

    @testset "2D getindex()/setindex! on MMatrix" begin
        sm = @MMatrix [1 3; 2 4]

        # Tuple, scalar
        @test (mm = MMatrix{2,2,Int}(); mm[(2,1),(2,1)] = sm[(2,1),(2,1)]; (@inferred getindex(mm, (2,1), (2,1)))::MMatrix == @MMatrix [4 2; 3 1])
        @test (mm = MMatrix{2,2,Int}(); mm[1,(1,2)] = sm[1,(1,2)]; (@inferred getindex(mm, 1, (1,2)))::MVector == @MVector [1,3])
        @test (mm = MMatrix{2,2,Int}(); mm[(1,2),1] = sm[(1,2),1]; (@inferred getindex(mm, (1,2), 1))::MVector == @MVector [1,2])

        # Colon
        @test (mm = MMatrix{2,2,Int}(); mm[:,:] = sm[:,:]; (@inferred getindex(mm, :, :))::MMatrix == @MMatrix [1 3; 2 4])
        @test (mm = MMatrix{2,2,Int}(); mm[(2,1),:] = sm[(2,1),:]; (@inferred getindex(mm, (2,1), :))::MMatrix == @MMatrix [2 4; 1 3])
        @test (mm = MMatrix{2,2,Int}(); mm[:,(2,1)] = sm[:,(2,1)]; (@inferred getindex(mm, :, (2,1)))::MMatrix == @MMatrix [3 1; 4 2])
        @test (mm = MMatrix{2,2,Int}(); mm[1,:] = sm[1,:]; (@inferred getindex(mm, 1, :))::MVector == @MVector [1,3])
        @test (mm = MMatrix{2,2,Int}(); mm[:,1] = sm[:,1]; (@inferred getindex(mm, :, 1))::MVector == @MVector [1,2])
    end

    @testset "2D getindex() with tuples on an Array" begin
        m = [1.0 2.0; 3.0 4.0]

        @test m[(1,2), (1,2)] === @SMatrix [1.0 2.0; 3.0 4.0]
        @test m[1, (1,2)] ===  (1.0, 2.0)
        @test m[(1,2), 1] ===  (1.0, 3.0)
    end

    @testset "3D scalar indexing" begin
        sa = SArray{(2,2,2), Int}([i*j*k for i = 1:2, j = 2:3, k=3:4])

        @test sa[1,1,2] === 8
        @test sa[1,2,1] === 9
        @test sa[2,1,1] === 12

        ma = MArray{(2,2,2), Int}()
        @test (ma[1,1,2] = 8; ma[1,1,2] === 8)
        @test (ma[1,2,1] = 9; ma[1,2,1] === 9)
        @test (ma[2,1,1] = 12; ma[2,1,1] === 12)
    end

    @testset "4D scalar indexing" begin
        sa = SArray{(2,2,2,2), Int}([i*j*k*l for i = 1:2, j = 2:3, k=3:4, l=4:5])

        @test sa[1,1,1,2] === 30
        @test sa[1,1,2,1] === 32
        @test sa[1,2,1,1] === 36
        @test sa[2,1,1,1] === 48

        ma = MArray{(2,2,2,2), Int}()
        @test (ma[1,1,1,2] = 30; ma[1,1,1,2] === 30)
        @test (ma[1,1,2,1] = 32; ma[1,1,2,1] === 32)
        @test (ma[1,2,1,1] = 36; ma[1,2,1,1] === 36)
        @test (ma[2,1,1,1] = 48; ma[2,1,1,1] === 48)
    end

end
