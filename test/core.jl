@testset "Core definitions and constructors" begin
    @testset "Inner Constructors" begin
        # SVector
        @test SVector{1,Int}((1,)).data === (1,)
        @test SVector{1,Float64}((1,)).data === (1.0,)
        @test_throws Exception SVector{1,Int}()

        @test SMatrix{1,1,Int}((1,)).data === (1,)
        @test SMatrix{1,1,Float64}((1,)).data === (1.0,)
        @test_throws Exception SMatrix{1,Int}()

        #@test MArray{(1,),Int,1,Tuple{Int}}((1,)).data === (1,)
        #@test MArray{(1,),Float64,1,Tuple{Float64}}((1,)).data === (1.0,)

        # default constructors
        #@test try; MArray{(1,),Int,1,Tuple{Int}}(); true; catch; false; end
    end

    @testset "Type parameter errors" begin
        # (not sure what type of exception these should be?)
        @test_throws Exception SVector{1.0,Int}((1,))
        @test_throws DimensionMismatch("No precise constructor for $(SVector{2, Int}) found. Length of input was 1.") SVector{2,Int}((1,))
        @test_throws Exception SVector{1,3}((1,))

        @test_throws Exception SMatrix{1.0,1,Int,1}((1,))
        @test_throws Exception SMatrix{1,1.0,Int,1}((1,))
        @test_throws Exception SMatrix{2,1,Int,1}((1,))
        @test_throws Exception SMatrix{1,2,Int,1}((1,))
        @test_throws Exception SMatrix{1,1,3,1}((1,))
        @test_throws Exception SMatrix{1,1,Int,2}((1,))

    end
#=
    @testset "Outer constructors" begin
        @test SArray{(1,),Int,1}((1,)).data === (1,)
        @test_inferred SArray{(1,),Int,1}((1,))
        @test SArray{(1,),Int}((1,)).data === (1,)
        @test_inferred SArray{(1,),Int}((1,))
        @test SArray{(1,)}((1,)).data === (1,)
        @test_inferred SArray{(1,)}((1,))

        @test MArray{(1,),Int,1}((1,)).data === (1,)
        @test_inferred MArray{(1,),Int,1}((1,))
        @test_inferred MArray{(1,),Int,1}()
        @test MArray{(1,),Int}((1,)).data === (1,)
        @test_inferred MArray{(1,),Int}((1,))
        @test_inferred MArray{(1,),Int}()
        @test MArray{(1,)}((1,)).data === (1,)
        @test_inferred MArray{(1,)}((1,))
    end

    @testset "Constructors that convert types" begin
        @test SArray{(1,),Float64,1}((1,)).data === (1.0,)
        @test_inferred SArray{(1,),Float64,1}((1,))
        @test SArray{(1,),Float64}((1,)).data === (1.0,)
        @test_inferred SArray{(1,),Float64}((1,))

        @test MArray{(1,),Float64,1}((1,)).data === (1.0,)
        @test_inferred MArray{(1,),Float64,1}((1,))
        @test MArray{(1,),Float64}((1,)).data === (1.0,)
        @test_inferred MArray{(1,),Float64}((1,))
    end
    =#
    @testset "eltype conversion" begin
        sa_int = SArray{Tuple{1}}((1,))
        ma_int = MArray{Tuple{1}}((1,))

        sa_float = SArray{Tuple{1}}((1.0,))
        ma_float = MArray{Tuple{1}}((1.0,))

        @test @inferred(convert(SArray{Tuple{1},Float64}, sa_int)) === sa_float
        @test @inferred(convert(SArray{Tuple{1},Float64,1}, sa_int)) === sa_float
        @test @inferred(convert(SArray{Tuple{1},Float64,1,1}, sa_int)) === sa_float

        @test @inferred(convert(MArray{Tuple{1},Float64}, ma_int)) == ma_float
        @test @inferred(convert(MArray{Tuple{1},Float64,1}, ma_int)) == ma_float
        @test @inferred(convert(MArray{Tuple{1},Float64,1,1}, ma_int)) == ma_float
    end

    @testset "StaticArray conversion" begin
        sa_int = SArray{Tuple{1}}((1,))
        ma_int = MArray{Tuple{1}}((1,))

        sa_float = SArray{Tuple{1}}((1.0,))
        ma_float = MArray{Tuple{1}}((1.0,))

        # SArray -> MArray
        #@test @inferred(convert(MArray, sa_int)) == ma_int
        @test @inferred(convert(MArray{Tuple{1}}, sa_int)) == ma_int
        @test @inferred(convert(MArray{Tuple{1},Int}, sa_int)) == ma_int
        @test @inferred(convert(MArray{Tuple{1},Int,1}, sa_int)) == ma_int
        @test @inferred(convert(MArray{Tuple{1},Int,1,1}, sa_int)) == ma_int

        @test @inferred(convert(MArray{Tuple{1},Float64}, sa_int)) == ma_float
        @test @inferred(convert(MArray{Tuple{1},Float64,1}, sa_int)) == ma_float
        @test @inferred(convert(MArray{Tuple{1},Float64,1,1}, sa_int)) == ma_float

        # MArray -> SArray
        #@test @inferred(convert(SArray, ma_int)) === sa_int
        @test @inferred(convert(SArray{Tuple{1}}, ma_int)) === sa_int
        @test @inferred(convert(SArray{Tuple{1},Int}, ma_int)) === sa_int
        @test @inferred(convert(SArray{Tuple{1},Int,1}, ma_int)) === sa_int
        @test @inferred(convert(SArray{Tuple{1},Int,1,1}, ma_int)) === sa_int

        @test @inferred(convert(SArray{Tuple{1},Float64}, ma_int)) === sa_float
        @test @inferred(convert(SArray{Tuple{1},Float64,1}, ma_int)) === sa_float
        @test @inferred(convert(SArray{Tuple{1},Float64,1,1}, ma_int)) === sa_float

        # Self-conversion returns the matrix itself
        @test convert(MArray, ma_int) === ma_int
        @test convert(MArray, ma_float) === ma_float
    end

    @testset "AbstractArray conversion" begin
        sa = SArray{Tuple{2,2}, Int}((3, 4, 5, 6))
        ma = MArray{Tuple{2,2}, Int}((3, 4, 5, 6))
        a = [3 5; 4 6]

        @test @inferred(convert(SArray{Tuple{2,2}}, a)) === sa
        @test @inferred(convert(SArray{Tuple{2,2},Int}, a)) === sa
        @test @inferred(convert(SArray{Tuple{2,2},Int,2}, a)) === sa
        @test @inferred(convert(SArray{Tuple{2,2},Int,2,4}, a)) === sa

        @test @inferred(convert(MArray{Tuple{2,2}}, a)) == ma
        @test @inferred(convert(MArray{Tuple{2,2},Int}, a)) == ma
        @test @inferred(convert(MArray{Tuple{2,2},Int,2}, a)) == ma
        @test @inferred(convert(MArray{Tuple{2,2},Int,2,4}, a)) == ma

        @test @inferred(convert(Array, sa)) == a
        @test @inferred(convert(Array{Int}, sa)) == a
        @test @inferred(convert(Array{Int,2}, sa)) == a

        @test @inferred(convert(Array, ma)) == a
        @test @inferred(convert(Array{Int}, ma)) == a
        @test @inferred(convert(Array{Int,2}, ma)) == a

        try
            convert(SVector, [1,2,3])
        catch err
            @test isa(err, ErrorException)
            @test startswith(err.msg, "The size of type")
        end
        @test_throws DimensionMismatch("expected input array of length 2, got length 3") convert(SVector{2}, [1,2,3])
    end
    @test_throws Exception Length{2.5}()
    @test Length(2) == Length{2}()
    @test Tuple{2, 3, 5} != StaticArrays.Size{(2, 3, 4)}
    @test StaticArrays.Size{(2, 3, 4)} != Tuple{2, 3, 5}
    @test StaticArrays.check_length(2) == nothing
    @test StaticArrays.check_length(StaticArrays.Dynamic()) == nothing

    @testset "Size" begin
        @test Size(zero(SMatrix{2, 3})) == Size(2, 3)
        @test Size(Transpose(zero(SMatrix{2, 3}))) == Size(3, 2)
        @test Size(Adjoint(zero(SMatrix{2, 3}))) == Size(3, 2)
        @test Size(Diagonal(SVector(1, 2, 3))) == Size(3, 3)
        @test Size(Transpose(Diagonal(SVector(1, 2, 3)))) == Size(3, 3)
        @test Size(UpperTriangular(zero(SMatrix{2, 2}))) == Size(2, 2)
        @test Size(LowerTriangular(zero(SMatrix{2, 2}))) == Size(2, 2)
        @test Size(LowerTriangular(Symmetric(zero(SMatrix{2, 2})))) == Size(2,2)
    end

    @testset "dimmatch" begin
        @test StaticArrays.dimmatch(3, 3)
        @test StaticArrays.dimmatch(3, StaticArrays.Dynamic())
        @test StaticArrays.dimmatch(StaticArrays.Dynamic(), 3)
        @test StaticArrays.dimmatch(StaticArrays.Dynamic(), StaticArrays.Dynamic())

        @test !StaticArrays.dimmatch(3, 2)
    end

    @testset "sizematch" begin
        @test StaticArrays.sizematch(Size(1, 2, 3), Size(1, 2, 3))
        @test StaticArrays.sizematch(Size(1, StaticArrays.Dynamic(), 3), Size(1, 2, 3))
        @test StaticArrays.sizematch(Size(1, 2, 3), Size(1, StaticArrays.Dynamic(), 3))
        @test StaticArrays.sizematch(Size(StaticArrays.Dynamic()), Size(StaticArrays.Dynamic()))

        @test !StaticArrays.sizematch(Size(1, 2, 3), Size(1, 2, 4))
        @test !StaticArrays.sizematch(Size(2, 2, 3), Size(1, 2, 3))
        @test !StaticArrays.sizematch(Size(2, 2, StaticArrays.Dynamic()), Size(1, 2, 3))
        @test !StaticArrays.sizematch(Size(1, 2), Size(1, 2, 3))
        @test !StaticArrays.sizematch(Size(1, 2, 3), Size(1, 2))

        sa = SArray{Tuple{2,3}, Int}((3, 4, 5, 6, 7, 8))
        a = [3 5 7; 4 6 8]

        @test StaticArrays.sizematch(Size(2, 3), sa)
        @test StaticArrays.sizematch(Size(2, StaticArrays.Dynamic()), sa)
        @test StaticArrays.sizematch(Size(2, 3), a)
        @test StaticArrays.sizematch(Size(2, StaticArrays.Dynamic()), a)

        @test !StaticArrays.sizematch(Size(2, 2), sa)
        @test !StaticArrays.sizematch(Size(3, StaticArrays.Dynamic()), sa)
        @test !StaticArrays.sizematch(Size(2, 2), a)
        @test !StaticArrays.sizematch(Size(3, StaticArrays.Dynamic()), a)
    end

    @testset "SOneTo" begin
        a = Base.OneTo(3)
        b = StaticArrays.SOneTo{2}()
        @test @inferred(promote(a, b)) === (a, Base.OneTo(2))
        @test @inferred(promote(b, a)) === (Base.OneTo(2), a)

        @test StaticArrays.SOneTo{2}(1:2) === StaticArrays.SOneTo{2}()
        @test convert(StaticArrays.SOneTo{2}, 1:2) === StaticArrays.SOneTo{2}()
        @test_throws DimensionMismatch StaticArrays.SOneTo{2}(1:3)
        @test_throws DimensionMismatch StaticArrays.SOneTo{2}(1:1)

        @test @inferred(intersect(SOneTo(2), SOneTo(4))) === SOneTo(2)
        @test @inferred(union(SOneTo(2), SOneTo(4))) === SOneTo(4)
    end
end
