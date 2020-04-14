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
    @test parent(A) === A
    @test parent(At) === A
    @test size(parent(At)) == (N1,N2)
    @test parent(b') === b
    @test size(parent(b')) == (N2,)

    # TSize
    ta = StaticArrays.TSize(A)
    @test !StaticArrays.istranpose(ta)
    @test size(ta) == (N1,N2)
    @test Size(ta) == Size(N1,N2)
    ta = StaticArrays.TSize(At)
    @test StaticArrays.istranpose(ta)
    @test size(ta) == (N2,N1)
    @test Size(ta) == Size(N2,N1)
    tb = StaticArrays.TSize(b')
    @test StaticArrays.istranpose(tb)
    ta = transpose(ta)
    @test !StaticArrays.istranpose(ta)
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

    @test_noalloc mul!(c,A,b)
    bmark = @benchmark mul!($c,$A,$b,$α,$β) samples=10 evals=10
    @test minimum(bmark).allocs == 0
    # @test_noalloc mul!(c, A, b, α, β)  # records 32 bytes
    bmark = @benchmark mul!($b,Transpose($A),$c) samples=10 evals=10
    @test minimum(bmark).allocs == 0
    # @test_noalloc mul!(b, Transpose(A), c)  # records 16 bytes
    bmark = @benchmark mul!($b,Transpose($A),$c,$α,$β) samples=10 evals=10
    @test minimum(bmark).allocs == 0
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

    b = @benchmark mul!($C,$a,$b') samples=10 evals=10
    @test minimum(b).allocs == 0
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
    @test minimum(b).allocs == 0
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
    @test minimum(b).allocs == 0
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
    @test minimum(b).allocs == 0
    # @test_noalloc mul!(C, Transpose(A), Transpose(B), α, β)  # records 64 bytes

    # Transpose Output
    C = rand(Mat{N1,N2})
    mul!(Transpose(C),Transpose(A),Transpose(B))
    @test C' ≈ A'B'
    b = @benchmark mul!(Transpose($C),Transpose($A),Transpose($B),$α,$β) samples=10 evals=10
    @test minimum(b).allocs == 0
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
