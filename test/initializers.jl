@testset "Initialization with SA" begin

SA_test_ref(x)   = SA[1,x,x]
SA_test_ref(x,T) = SA{T}[1,x,x]
@test @inferred(SA_test_ref(2))   === SVector{3,Int}((1,2,2))
@test @inferred(SA_test_ref(2.0)) === SVector{3,Float64}((1,2,2))
@test @inferred(SA_test_ref(2,Float32))   === SVector{3,Float32}((1,2,2))

SA_test_vcat(x)   = SA[1;x;x]
SA_test_vcat(x,T) = SA{T}[1;x;x]
@test @inferred(SA_test_vcat(2))   === SVector{3,Int}((1,2,2))
@test @inferred(SA_test_vcat(2.0)) === SVector{3,Float64}((1,2,2))
@test @inferred(SA_test_vcat(2,Float32))   === SVector{3,Float32}((1,2,2))

SA_test_hcat(x)   = SA[1 x x]
SA_test_hcat(x,T) = SA{T}[1 x x]
@test @inferred(SA_test_hcat(2))   === SMatrix{1,3,Int}((1,2,2))
@test @inferred(SA_test_hcat(2.0)) === SMatrix{1,3,Float64}((1,2,2))
@test @inferred(SA_test_hcat(2,Float32))   === SMatrix{1,3,Float32}((1,2,2))

# hvcat needs to be in a function for the row argument to constant propagate
SA_test_hvcat(x) = SA[1 x x;
                      x 2 x]
SA_test_hvcat(x,T) = SA{T}[1 x x;
                           x 2 x]
@test @inferred(SA_test_hvcat(3))   === SMatrix{2,3,Int}((1,3,3,2,3,3))
@test @inferred(SA_test_hvcat(3.0)) === SMatrix{2,3,Float64}((1,3,3,2,3,3))
@test @inferred(SA_test_hvcat(1.0im)) === SMatrix{2,3,ComplexF64}((1,1im,1im,2,1im,1im))
@test @inferred(SA_test_hvcat(3,Float32))   === SMatrix{2,3,Float32}((1,3,3,2,3,3))

@test SA[1] === SVector{1,Int}((1))

@test_throws ArgumentError("SA[...] matrix rows of length (3, 2) are inconsistent") SA[1 2 3;
                                                                                       4 5]
@test_throws ArgumentError("SA[...] matrix rows of length (2, 3) are inconsistent") SA[1 2;
                                                                                       3 4 5]
@test SA_F64[1, 2] === SVector{2,Float64}((1,2))
@test SA_F32[1, 2] === SVector{2,Float32}((1,2))

@test_inlined SA[1,2]
@test_inlined SA[1 2]
@test_inlined SA[1;2]
@test_inlined SA_test_hvcat(3)

SA_test_hvncat1(x) = SA[1 x;x 2;;;x 2;1 x]
SA_test_hvncat2(x) = SA[1;x;;x;2;;;x;2;;1;x]
if VERSION >= v"1.7.0"
    @test SA[1;;;2] === SArray{Tuple{1,1,2}}(1,2)
    @test SA[1;2;;1;2] === SMatrix{2,2}(1,2,1,2)
    @test SA[1 2;1 2 ;;; 1 2;1 2] === SArray{Tuple{2,2,2}}(Tuple([1 2;1 2 ;;; 1 2;1 2]))
    @test_inlined SA_test_hvncat1(3)
    @test_inlined SA_test_hvncat2(2)
    if VERSION < v"1.10-DEV"
        @test_throws ArgumentError SA[1;2;;3]
    else
        @test_throws DimensionMismatch SA[1;2;;3]
    end
end

# https://github.com/JuliaArrays/StaticArrays.jl/pull/685
@test Union{}[] isa Vector{Union{}}
@test Base.typed_vcat(Union{}) isa Vector{Union{}}
@test Base.typed_hcat(Union{}) isa Vector{Union{}}
@test Base.typed_hvcat(Union{}, ()) isa Vector{Union{}}
@test_throws Union{MethodError, ArgumentError} Union{}[1]
@test_throws Union{MethodError, ArgumentError} Union{}[1 2]
@test_throws Union{MethodError, ArgumentError} Union{}[1; 2]
@test_throws Union{MethodError, ArgumentError} Union{}[1 2; 3 4]

end
