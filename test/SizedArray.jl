@testset "SizedArray" begin
    @testset "Inner Constructors" begin
        @test SizedArray{Tuple{2}, Int, 1}((3, 4)).data == [3, 4]
        @test SizedArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
        @test SizedArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
        @test size(SizedArray{Tuple{4, 5}, Int, 2}().data) == (4, 5)
        @test size(SizedArray{Tuple{4, 5}, Int}().data) == (4, 5)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception SizedArray{Tuple{1},Int,2}()
        @test_throws Exception SArray{Tuple{3, 4},Int,1}()

        # Parameter/input size mismatch
        @test_throws Exception SizedArray{Tuple{1},Int,2}([2; 3])
        @test_throws Exception SizedArray{Tuple{1},Int,2}((2, 3))
    end

    @testset "Outer Constructors" begin
        # From Array
        @test @inferred(SizedArray{Tuple{2},Float64,1}([1,2]))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2},Float64}([1,2]))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2}}([1,2]))::SizedArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedArray{Tuple{2,2}}([1 2;3 4]))::SizedArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # From Array, reshaped
        @test @inferred(SizedArray{Tuple{2,2}}([1,2,3,4]))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        # Uninitialized
        @test @inferred(SizedArray{Tuple{2,2},Int,2}()) isa SizedArray{Tuple{2,2},Int,2,2}
        @test @inferred(SizedArray{Tuple{2,2},Int}()) isa SizedArray{Tuple{2,2},Int,2,2}

        # From Tuple
        @test @inferred(SizedArray{Tuple{2},Float64,1}((1,2)))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2},Float64}((1,2)))::SizedArray{Tuple{2},Float64,1,1} == [1.0, 2.0]
        @test @inferred(SizedArray{Tuple{2}}((1,2)))::SizedArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedArray{Tuple{2,2}}((1,2,3,4)))::SizedArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
    end

    @testset "SizedVector and SizedMatrix" begin
        @test @inferred(SizedVector{2}([1,2]))::SizedArray{Tuple{2},Int,1,1} == [1,2]
        @test @inferred(SizedVector{2}((1,2)))::SizedArray{Tuple{2},Int,1,1} == [1,2]
        # Reshaping
        @test @inferred(SizedVector{2}([1 2]))::SizedArray{Tuple{2},Int,1,2} == [1,2]
        # Back to Vector
        @test Vector(SizedVector{2}((1,2))) == [1,2]
        @test convert(Vector, SizedVector{2}((1,2))) == [1,2]

        @test @inferred(SizedMatrix{2,2}([1 2; 3 4]))::SizedArray{Tuple{2,2},Int,2,2} == [1 2; 3 4]
        # Reshaping
        @test @inferred(SizedMatrix{2,2}([1,2,3,4]))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test @inferred(SizedMatrix{2,2}((1,2,3,4)))::SizedArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
        # Back to Matrix
        @test Matrix(SizedMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
        @test convert(Matrix, SizedMatrix{2,2}([1 2;3 4])) == [1 2; 3 4]
    end

    # setindex
    sa = SizedArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]

    if isdefined(Base, :mightalias) # v0.7-
        @testset "aliasing" begin
            a1 = rand(4)
            a2 = copy(a1)
            sa1 = SizedVector{4}(a1)
            sa2 = SizedVector{4}(a2)
            @test Base.mightalias(a1, sa1)
            @test Base.mightalias(sa1, SizedVector{4}(a1))
            @test !Base.mightalias(a2, sa1)
            @test !Base.mightalias(sa1, SizedVector{4}(a2))
            @test Base.mightalias(sa1, view(sa1, 1:2))
            @test Base.mightalias(a1, view(sa1, 1:2))
            @test Base.mightalias(sa1, view(a1, 1:2))
        end
    end

    @testset "back to Array" begin
        @test Array(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int}(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int, 1}(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Vector(SizedArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test convert(Vector, SizedArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Array, SizedArray{Tuple{2,2,2,2}, Int}(ones(2,2,2,2))) == ones(2,2,2,2)
        # Conversion after reshaping
        @test_broken Array(SizedMatrix{2,2}([1,2,3,4])) == [1 3; 2 4]
    end

    @testset "promotion" begin
        @test @inferred(promote_type(SizedVector{1,Float64,1}, SizedVector{1,BigFloat,1})) == SizedVector{1,BigFloat,1}
        @test @inferred(promote_type(SizedVector{2,Int,1}, SizedVector{2,Float64,1})) === SizedVector{2,Float64,1}
        @test @inferred(promote_type(SizedMatrix{2,3,Float32,2}, SizedMatrix{2,3,Complex{Float64},2})) === SizedMatrix{2,3,Complex{Float64},2}
    end
end
