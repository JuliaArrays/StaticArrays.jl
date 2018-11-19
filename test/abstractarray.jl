using StaticArrays, Test, LinearAlgebra

@testset "AbstractArray interface" begin
    @testset "size and length" begin
        m = @SMatrix [1 2 3; 4 5 6; 7 8 9; 10 11 12]

        @test length(m) == 12
        @test IndexStyle(m) == IndexLinear()
        @test Base.isassigned(m, 2, 2) == true
    end

    @testset "similar_type" begin
        @test @inferred(similar_type(SVector{3,Int})) == SVector{3,Int}
        @test @inferred(similar_type(@SVector [1,2,3])) == SVector{3,Int}

        @test @inferred(similar_type(SVector{3,Int}, Float64)) == SVector{3,Float64}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Size(2))) == SVector{2, Int}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64, Size(2))) == SVector{2, Float64}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64, Size(2))) == SVector{2, Float64}

        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64)) == SMatrix{3, 3, Float64, 9}
        @test @inferred(similar_type(SVector{2,Int}, Size(3,3))) == SMatrix{3, 3, Int, 9}
        @test @inferred(similar_type(SVector{2,Int}, Float64, Size(3,3))) == SMatrix{3, 3, Float64, 9}

        @test @inferred(similar_type(SArray{Tuple{4,4,4},Int,3,64}, Float64)) == SArray{Tuple{4,4,4}, Float64, 3, 64}
        @test @inferred(similar_type(SVector{2,Int}, Size(3,3,3))) == SArray{Tuple{3,3,3}, Int, 3, 27}
        @test @inferred(similar_type(SVector{2,Int}, Float64, Size(3,3,3))) == SArray{Tuple{3,3,3}, Float64, 3, 27}

        # Some specializations for the mutable case
        @test @inferred(similar_type(MVector{3,Int}, Float64)) == MVector{3,Float64}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Size(2))) == MVector{2, Int}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64, Size(2))) == MVector{2, Float64}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64, Size(2))) == MVector{2, Float64}

        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64)) == MMatrix{3, 3, Float64, 9}
        @test @inferred(similar_type(MVector{2,Int}, Size(3,3))) == MMatrix{3, 3, Int, 9}
        @test @inferred(similar_type(MVector{2,Int}, Float64, Size(3,3))) == MMatrix{3, 3, Float64, 9}

        @test @inferred(similar_type(MArray{Tuple{4,4,4},Int,3,64}, Float64)) == MArray{Tuple{4,4,4}, Float64, 3, 64}
        @test @inferred(similar_type(MVector{2,Int}, Size(3,3,3))) == MArray{Tuple{3,3,3}, Int, 3, 27}
        @test @inferred(similar_type(MVector{2,Int}, Float64, Size(3,3,3))) == MArray{Tuple{3,3,3}, Float64, 3, 27}
    end

    @testset "similar" begin
        sv = @SVector [1,2,3]
        sm = @SMatrix [1 2; 3 4]
        sa = SArray{Tuple{1,1,1},Int,3,1}((1,))

        @test isa(@inferred(similar(sv)), MVector{3,Int})
        @test isa(@inferred(similar(sv, Float64)), MVector{3,Float64})
        @test isa(@inferred(similar(sv, Size(4))), MVector{4,Int})
        @test isa(@inferred(similar(sv, Float64, Size(4))), MVector{4,Float64})

        @test isa(@inferred(similar(sm)), MMatrix{2,2,Int,4})
        @test isa(@inferred(similar(sm, Float64)), MMatrix{2,2,Float64,4})
        @test isa(@inferred(similar(sv, Size(3,3))), MMatrix{3,3,Int,9})
        @test isa(@inferred(similar(sv, Float64, Size(3,3))), MMatrix{3,3,Float64,9})

        @test isa(@inferred(similar(sa)), MArray{Tuple{1,1,1},Int,3,1})
        @test isa(@inferred(similar(SArray{Tuple{1,1,1},Int,3,1})), MArray{Tuple{1,1,1},Int,3,1})
        @test isa(@inferred(similar(sa, Float64)), MArray{Tuple{1,1,1},Float64,3,1})
        @test isa(@inferred(similar(SArray{Tuple{1,1,1},Int,3,1}, Float64)), MArray{Tuple{1,1,1},Float64,3,1})
        @test isa(@inferred(similar(sv, Size(3,3,3))), MArray{Tuple{3,3,3},Int,3,27})
        @test isa(@inferred(similar(sv, Float64, Size(3,3,3))), MArray{Tuple{3,3,3},Float64,3,27})

        @test isa(@inferred(similar(Diagonal{Int}, Size(2,2))), MArray{Tuple{2, 2}, Int, 2, 4})
        @test isa(@inferred(similar(SizedArray, Int, Size(2,2))), SizedArray{Tuple{2, 2}, Int, 2, 2})
        @test isa(@inferred(similar(Matrix{Int}, Int, Size(2,2))), SizedArray{Tuple{2, 2}, Int, 2, 2})
    end

    @testset "similar and Base.Slice (issue #548)" begin
        v = @SVector [1]
        b = similar(v, Int, Base.Slice(SOneTo(3)))
        @test b isa MVector{3,Int}
    end

    @testset "reshape" begin
        @test @inferred(reshape(SVector(1,2,3,4), Size(2,2))) === SMatrix{2,2}(1,2,3,4)
        @test @inferred(reshape([1,2,3,4], Size(2,2)))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]

        @test @inferred(vec(SMatrix{2, 2}([1 2; 3 4])))::SVector{4,Int} == [1, 3, 2, 4]

        # AbstractArray
        # CartesianIndex
        @test reshape(view(ones(4, 4), 1:4, 1:2), Size(4, 2)) == SMatrix{4,2}(ones(4, 2))
        # IndexLinear
        @test reshape(view(ones(4, 4), 1, 1:4), Size(4, 1)) == SMatrix{4,1}(ones(4, 1))
        @test_throws DimensionMismatch reshape(view(ones(4,4), 1:4, 1:2), Size(5, 2))
    end

    @testset "copy" begin
        @test @inferred(copy(SMatrix{2, 2}([1 2; 3 4]))) === @SMatrix [1 2; 3 4]
        @test @inferred(copy(MMatrix{2, 2}([1 2; 3 4])))::MMatrix == [1 2; 3 4]
    end
end
