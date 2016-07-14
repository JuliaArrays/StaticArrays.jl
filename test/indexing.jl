@testset "Indexing" begin
    @testset "Linear getindex() on SVector" begin
        vec = [4,5,6,7]
        sv = SVector{4}(vec)

        # Tuple
        @test (@inferred getindex(sv, (4,3,2,1))) === SVector((7,6,5,4))

        # Colon
        @test (@inferred getindex(sv,:)) === sv
    end

    @testset "Linear getindex() on SVector" begin
        vec = [4,5,6,7]
        mv = MVector{4,Int}(vec)

        # Tuple
        @test (mv[(1,2,3,4)] = vec; (@inferred getindex(mv, (4,3,2,1)))::MVector{4,Int} == MVector((7,6,5,4)))

        # Colon
        @test (mv[:] = vec; (@inferred getindex(mv, :))::MVector{4,Int} == MVector((4,5,6,7)))
    end

    @testset "2D getindex() on SVector" begin
        sm = @SMatrix [1 3; 2 4]

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

    #=
    @testset "Linear getindex() on SMatrix" begin
        mat = reshape([4,5,6,7], (2,2))
        sm = SMatrix{2,2}(mat)

        # Scalar
        @test sm[1] == mat[1]
        @test sm[2] == mat[2]
        @test sm[3] == mat[3]
        @test sm[4] == mat[4]
        @test_inferred getindex(sm, 1)

        # Tuple
        @test sm[(4,3,2,1)] === SVector((7,6,5,4))
        @test_inferred getindex(sm, (4,3,2,1))

        # Colon
        @test sm[:] === SVector{4}(mat[:])
        @test_inferred getindex(sm, :)

    end
    =#
#=
    @testset "Multi-dimensional getindex()" begin
        a = reshape([1,2,3,4,5,6,7,8],(2,2,2))
        sa = SArray{(2,2,2)}(a)
        ma = MArray{(2,2,2)}(a)

        # Scalar
        @test sa[1,1,1] == 1
        @test sa[2,1,1] == 2
        @test sa[1,2,1] == 3
        @test sa[2,2,1] == 4
        @test sa[1,1,2] == 5
        @test sa[2,1,2] == 6
        @test sa[1,2,2] == 7
        @test sa[2,2,2] == 8
        @test_inferred getindex(sa, 1, 1, 1)

        @test ma[1,1,1] == 1
        @test ma[2,1,1] == 2
        @test ma[1,2,1] == 3
        @test ma[2,2,1] == 4
        @test ma[1,1,2] == 5
        @test ma[2,1,2] == 6
        @test ma[1,2,2] == 7
        @test ma[2,2,2] == 8
        @test_inferred getindex(ma, 1, 1, 1)

        # Colon
        @test sa[:,:,:] === sa
        @test_inferred getindex(sa, :, :, :)
        @test ma[:,:,:] == ma
        @test isa(ma[:,:,:], MArray{(2,2,2)})
        @test_inferred getindex(ma, :, :, :)

        # Mixed
        @test sa[:,(2,1),1] === SArray{(2,2)}(a[:,[2,1],1])
        @test ma[:,(2,1),1] == MArray{(2,2)}(a[:,[2,1],1])
        @test_inferred getindex(sa,:,(2,1),1)
        @test isa(ma[:,(2,1),1], MArray{(2,2)})
        @test_inferred getindex(ma,:,(2,1),1)
    end

    @testset "Linear setindex!()" begin
        mat = reshape([4,5,6,7], (2,2))

        # Scalar
        mm = MMatrix{(2,2)}(zeros(Int,2,2))
        mm[1] = mat[1]
        mm[2] = mat[2]
        mm[3] = mat[3]
        mm[4] = mat[4]

        @test mm == MMatrix{(2,2)}(mat)

        # Vector
        mm = MMatrix{(2,2)}(zeros(Int,2,2))
        mm[[4,3,2,1]] = mat[[4,3,2,1]]

        @test mm == MMatrix{(2,2)}(mat)

        # Tuple
        mm = MMatrix{(2,2)}(zeros(Int,2,2))
        mm[(4,3,2,1)] = mat[[4,3,2,1]]

        @test mm == MMatrix{(2,2)}(mat)

        # Colon
        mm = MMatrix{(2,2)}(zeros(Int,2,2))
        mm[:] = mat[:]

        @test mm == MMatrix{(2,2)}(mat)
    end

    @testset "Multi-dimensional setindex!()" begin
        a = reshape([1,2,3,4,5,6,7,8],(2,2,2))

        # Scalar
        ma = MArray{(2,2,2)}(zeros(Int,2,2,2))
        ma[1,1,1] = a[1,1,1]
        ma[2,1,1] = a[2,1,1]
        ma[1,2,1] = a[1,2,1]
        ma[2,2,1] = a[2,2,1]
        ma[1,1,2] = a[1,1,2]
        ma[2,1,2] = a[2,1,2]
        ma[1,2,2] = a[1,2,2]
        ma[2,2,2] = a[2,2,2]
        @test ma == a

        # Colon
        ma = MArray{(2,2,2)}(zeros(Int,2,2,2))
        ma[:,:,:] = a
        @test ma == a

        # Mixed
        ma = MArray{(2,2,2)}(zeros(Int,2,2,2))
        ma[:,(2,1),1] = a[:,[2,1],1]
        ma[:,[2,1],2] = a[:,[2,1],2]
        @test ma == a
    end=#
end
