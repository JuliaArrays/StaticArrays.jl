using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Test

macro test_noalloc(ex)
    esc(quote
        $ex
        @test(@allocated($ex) == 0)
    end)
end

mul_add_wrappers = [
    m -> m,
    m -> Symmetric(m, :U),
    m -> Symmetric(m, :L),
    m -> Hermitian(m, :U),
    m -> Hermitian(m, :L),
    m -> UpperTriangular(m),
    m -> LowerTriangular(m),
    m -> UnitUpperTriangular(m),
    m -> UnitLowerTriangular(m),
    m -> Adjoint(m),
    m -> Transpose(m),
    m -> Diagonal(m)]


# check_dims
@test StaticArrays.check_dims(Size(4,), Size(4,3), Size(3,))
@test !StaticArrays.check_dims(Size(4,), Size(4,3), Size(4,))
@test !StaticArrays.check_dims(Size(4,), Size(3,4), Size(4,))

@test StaticArrays.check_dims(Size(4,), Size(4,3), Size(3,1))
@test StaticArrays.check_dims(Size(4,1), Size(4,3), Size(3,))
@test StaticArrays.check_dims(Size(4,1), Size(4,3), Size(3,1))
@test !StaticArrays.check_dims(Size(4,2), Size(4,3), Size(3,))
@test !StaticArrays.check_dims(Size(4,), Size(4,3), Size(3,2))
@test StaticArrays.check_dims(Size(4,2), Size(4,3), Size(3,2))
@test !StaticArrays.check_dims(Size(4,2), Size(4,3), Size(3,3))

function test_multiply_add(N1,N2,ArrayType=MArray)
    if ArrayType <: MArray
        Mat = MMatrix
        Vec = MVector
    elseif ArrayType <: SizedArray
        Mat = SizedMatrix
        Vec = SizedVector
    end
    α,β = 2.0, 1.0

    A = rand(Mat{N1,N2})
    At = Transpose(A)
    b = rand(Vec{N2})
    c = rand(Vec{N1})

    # Parent
    if !(ArrayType <: SizedArray)
        @test parent(A) === A
    end
    @test parent(At) === A
    @test size(parent(At)) == (N1,N2)
    @test parent(b') === b
    @test size(parent(b')) == (N2,)

    # TSize
    ta = StaticArrays.TSize(A)
    @test !StaticArrays.istranspose(ta)
    @test size(ta) == (N1,N2)
    @test Size(ta) == Size(N1,N2)
    ta = StaticArrays.TSize(At)
    @test StaticArrays.istranspose(ta)
    @test size(ta) == (N2,N1)
    @test Size(ta) == Size(N2,N1)
    tb = StaticArrays.TSize(b')
    @test StaticArrays.access_type(tb) === :adjoint
    ta = transpose(ta)
    @test !StaticArrays.istranspose(ta)
    @test size(ta) == (N1,N2)
    @test Size(ta) == Size(N1,N2)

    # A*b
    mul!(c,A,b)
    @test c ≈ A*b
    mul!(c,A,b,1.0,0.0)
    @test c ≈ A*b
    mul!(c,A,b,1.0,1.0)
    @test c ≈ 2A*b

    # matrix-transpose × vector
    mul!(b,At,c)
    @test b ≈ A'c
    mul!(b,At,c,2.0,0.0)
    @test b ≈ 2A'c
    mul!(b,At,c,1.0,2.0)
    @test b ≈ 5A'c

    if !(ArrayType <: SizedArray)
        @test_noalloc mul!(c,A,b)
    else
        mul!(c,A,b)
        @test_broken(@allocated(mul!(c,A,b)) == 0)
    end
    expected_transpose_allocs = 0
    bmark = @benchmark mul!($c,$A,$b,$α,$β) samples=10 evals=10
    @test minimum(bmark).allocs == 0
    # @test_noalloc mul!(c, A, b, α, β)  # records 32 bytes
    bmark = @benchmark mul!($b,Transpose($A),$c) samples=10 evals=10
    @test minimum(bmark).allocs <= expected_transpose_allocs
    # @test_noalloc mul!(b, Transpose(A), c)  # records 16 bytes
    bmark = @benchmark mul!($b,Transpose($A),$c,$α,$β) samples=10 evals=10
    @test minimum(bmark).allocs <= expected_transpose_allocs
    # @test_noalloc mul!(b, Transpose(A), c, α, β)  # records 48 bytes

    # outer product
    C = rand(Mat{N1,N2})
    a = rand(Vec{N1})
    b = rand(Vec{N2})
    mul!(C,a,b')
    @test C ≈ a*b'
    mul!(C,a,b',2.,0.)
    @test C ≈ 2a*b'
    mul!(C,a,b',1.,1.)
    @test C ≈ 3a*b'

    b = @benchmark mul!($C,$a,$(b')) samples=10 evals=10
    @test minimum(b).allocs <= expected_transpose_allocs
    # @test_noalloc mul!(C, a, b')  # records 16 bytes

    # A × B
    A = rand(Mat{N1,N2})
    B = rand(Mat{N2,N2})
    C = rand(Mat{N1,N2})
    mul!(C,A,B)
    @test C ≈ A*B
    mul!(C,A,B,2.0,0.0)
    @test C ≈ 2A*B
    mul!(C,A,B,2.0,1.0)
    @test C ≈ 4A*B

    b = @benchmark mul!($C,$A,$B,$α,$β) samples=10 evals=10
    @test minimum(b).allocs == 0
    # @test_noalloc mul!(C, A, B, α, β)  # records 32 bytes

    # A'B
    At = Transpose(A)
    mul!(B,At,C)
    @test B ≈ A'C
    mul!(B,At,C,2.0,0.0)
    @test B ≈ 2A'C
    mul!(B,At,C,2.0,1.0)
    @test B ≈ 4A'C

    b = @benchmark mul!($B,Transpose($A),$C,$α,$β) samples=10 evals=10
    @test minimum(b).allocs <= expected_transpose_allocs
    # @test_noalloc mul!(B, Transpose(A), C, α, β)  # records 48 bytes

    # A*B'
    Bt = Transpose(B)
    mul!(C,A,Bt)
    @test C ≈ A*B'
    mul!(C,A,Bt,2.0,0.0)
    @test C ≈ 2A*B'
    mul!(C,A,Bt,2.0,1.0)
    @test C ≈ 4A*B'

    b = @benchmark mul!($C,$A,Transpose($B),$α,$β) samples=10 evals=10
    @test minimum(b).allocs <= expected_transpose_allocs
    # @test_noalloc mul!(C, A, Transpose(B), α, β)  # records 48 bytes

    # A'B'
    B = rand(Mat{N1,N1})
    C = rand(Mat{N2,N1})
    mul!(C,Transpose(A),Transpose(B))
    @test C ≈ A'B'
    mul!(C,Transpose(A),Transpose(B),2.0,0.0)
    @test C ≈ 2A'B'
    mul!(C,Transpose(A),Transpose(B),2.0,1.0)
    @test C ≈ 4A'B'

    b = @benchmark mul!($C,Transpose($A),Transpose($B),$α,$β) samples=10 evals=10
    @test minimum(b).allocs <= 2*expected_transpose_allocs
    # @test_noalloc mul!(C, Transpose(A), Transpose(B), α, β)  # records 64 bytes

    # Transpose Output
    C = rand(Mat{N1,N2})
    mul!(Transpose(C),Transpose(A),Transpose(B))
    @test C' ≈ A'B'
    b = @benchmark mul!(Transpose($C),Transpose($A),Transpose($B),$α,$β) samples=10 evals=10
    @test minimum(b).allocs <= expected_transpose_allocs*3
    # @test_noalloc mul!(Transpose(C), Transpose(A), Transpose(B), α, β)  # records 80 bytes
end

# Test the three different
@testset "matrix multiply-add" begin
    test_multiply_add(3,4)
    test_multiply_add(5,6)
    test_multiply_add(15,16)
    test_multiply_add(3,4,SizedArray)
    test_multiply_add(5,6,SizedArray)
    test_multiply_add(15,16,SizedArray)
end

function test_wrappers_for_size(N, test_block)
    C = rand(MMatrix{N,N,Int})
    Cv = rand(MVector{N,Int})
    A = rand(SMatrix{N,N,Int})
    B = rand(SMatrix{N,N,Int})
    bv = rand(SVector{N,Int})
    # matrix-vector
    for wrapper in mul_add_wrappers
        mul!(Cv, wrapper(A), bv)
        @test Cv == wrapper(Array(A))*Array(bv)
    end

    # matrix-matrix
    for wrapper_c in [identity, Transpose], wrapper_a in mul_add_wrappers, wrapper_b in mul_add_wrappers
        mul!(wrapper_c(C), wrapper_a(A), wrapper_b(B))
        @test wrapper_c(C) == wrapper_a(Array(A))*wrapper_b(Array(B))
    end

    # block matrices
    if test_block

        C_block = rand(MMatrix{N,N,SMatrix{2,2,Int,4}})
        Cv_block = rand(MVector{N,SMatrix{2,2,Int,4}})
        A_block = rand(SMatrix{N,N,SMatrix{2,2,Int,4}})
        B_block = rand(SMatrix{N,N,SMatrix{2,2,Int,4}})
        bv_block = rand(SVector{N,SMatrix{2,2,Int,4}})
        
        # matrix-vector
        for wrapper in mul_add_wrappers
            # LinearAlgebra can't handle these
            if all(T -> !isa(wrapper([1 2; 3 4]), T), [Symmetric, Hermitian, Diagonal])
                mul!(Cv_block, wrapper(A_block), bv_block)
                @test Cv_block == wrapper(Array(A_block))*Array(bv_block)
            end
        end

        # matrix-matrix
        for wrapper_a in mul_add_wrappers, wrapper_b in mul_add_wrappers
            if all(T -> !isa(wrapper_a([1 2; 3 4]), T) && !isa(wrapper_b([1 2; 3 4]), T), [Symmetric, Hermitian, Diagonal])
                mul!(C_block, wrapper_a(A_block), wrapper_b(B_block))
                @test C_block == wrapper_a(Array(A_block))*wrapper_b(Array(B_block))
            end
        end
    end

end

@testset "Testing different wrappers" begin
    test_wrappers_for_size(2, true)
    test_wrappers_for_size(8, false)
    test_wrappers_for_size(16, false)
end
