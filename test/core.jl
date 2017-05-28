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
        @test_throws Exception SVector{2,Int}((1,))
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

    @testset "eltype conversion" begin
        sa_int = SArray{(1,)}((1,))
        ma_int = MArray{(1,)}((1,))

        sa_float = SArray{(1,)}((1.0,))
        ma_float = MArray{(1,)}((1.0,))

        @test convert(SArray{(1,),Float64}, sa_int) === sa_float
        @test_inferred convert(SArray{(1,),Float64}, sa_int)
        @test convert(SArray{(1,),Float64,1}, sa_int) === sa_float
        @test_inferred convert(SArray{(1,),Float64,1}, sa_int)
        @test convert(SArray{(1,),Float64,1,Tuple{Float64}}, sa_int) === sa_float
        @test_inferred convert(SArray{(1,),Float64,1,Tuple{Float64}}, sa_int)

        @test convert(MArray{(1,),Float64}, ma_int) == ma_float
        @test_inferred convert(MArray{(1,),Float64}, ma_int)
        @test convert(MArray{(1,),Float64,1}, ma_int) == ma_float
        @test_inferred convert(MArray{(1,),Float64,1}, ma_int)
        @test convert(MArray{(1,),Float64,1,Tuple{Float64}}, ma_int) == ma_float
        @test_inferred convert(MArray{(1,),Float64,1,Tuple{Float64}}, ma_int)
    end

    @testset "StaticArray conversion" begin
        sa_int = SArray{(1,)}((1,))
        ma_int = MArray{(1,)}((1,))

        sa_float = SArray{(1,)}((1.0,))
        ma_float = MArray{(1,)}((1.0,))

        # SArray -> MArray
        @test convert(MArray, sa_int) == ma_int
        @test_inferred convert(MArray, sa_int)
        @test convert(MArray{(1,)}, sa_int) == ma_int
        @test_inferred convert(MArray{(1,)}, sa_int)
        @test convert(MArray{(1,),Int}, sa_int) == ma_int
        @test_inferred convert(MArray{(1,),Int}, sa_int)
        @test convert(MArray{(1,),Int,1}, sa_int) == ma_int
        @test_inferred convert(MArray{(1,),Int,1}, sa_int)
        @test convert(MArray{(1,),Int,1,Tuple{Int}}, sa_int) == ma_int
        @test_inferred convert(MArray{(1,),Int,1,Tuple{Int}}, sa_int)

        @test convert(MArray{(1,),Float64}, sa_int) == ma_float
        @test_inferred convert(MArray{(1,),Float64}, sa_int)
        @test convert(MArray{(1,),Float64,1}, sa_int) == ma_float
        @test_inferred convert(MArray{(1,),Float64,1}, sa_int)
        @test convert(MArray{(1,),Float64,1,Tuple{Float64}}, sa_int) == ma_float
        @test_inferred convert(MArray{(1,),Float64,1,Tuple{Float64}}, sa_int)

        # MArray -> SArray
        @test convert(SArray, ma_int) === sa_int
        @test_inferred convert(SArray, ma_int)
        @test convert(SArray{(1,)}, ma_int) === sa_int
        @test_inferred convert(SArray{(1,)}, ma_int)
        @test convert(SArray{(1,),Int}, ma_int) === sa_int
        @test_inferred convert(SArray{(1,),Int}, ma_int)
        @test convert(SArray{(1,),Int,1}, ma_int) === sa_int
        @test_inferred convert(SArray{(1,),Int,1}, ma_int)
        @test convert(SArray{(1,),Int,1,Tuple{Int}}, ma_int) === sa_int
        @test_inferred convert(SArray{(1,),Int,1,Tuple{Int}}, ma_int)

        @test convert(SArray{(1,),Float64}, ma_int) === sa_float
        @test_inferred convert(SArray{(1,),Float64}, ma_int)
        @test convert(SArray{(1,),Float64,1}, ma_int) === sa_float
        @test_inferred convert(SArray{(1,),Float64,1}, ma_int)
        @test convert(SArray{(1,),Float64,1,Tuple{Float64}}, ma_int) === sa_float
        @test_inferred convert(SArray{(1,),Float64,1,Tuple{Float64}}, ma_int)
    end
    =#
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
    end
    @test_throws Exception Length{2.5}()
    @test Length(2) == Length{2}()
    @test Tuple{2, 3, 5} != Size{(2, 3, 4)}
    @test Size{(2, 3, 4)} != Tuple{2, 3, 5} 

end
