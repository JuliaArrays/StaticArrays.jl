
function test_muladd(a,b,c, α, β, test_transpose=true)
    c0 = copy(c)
    b0 = copy(b)

    mul!(c,a,b)
    @test c ≈ a*b

    c .= c0
    b .= b0
    mul!(c,a,b,α,β)
    @test c ≈ a*b*α + c0*β

    @test (@allocated mul!(c,a,b)) == 0
    @test (@allocated mul!(c,a,b,α,β)) == 0

    if test_transpose
        c .= c0
        b .= b0
        mul!(b, Transpose(a), c)
        @test b ≈ a'c

        c .= c0
        b .= b0
        mul!(b,Transpose(a),c,α,β)
        @test b ≈ a'c*α + b0*β

        @test (@allocated mul!(b,Transpose(a),c)) == 0
        @test (@allocated mul!(b,Transpose(a),c,α,β)) == 0
    end
end

@testset "Matrix multiply-add" begin
    α,β = 2.0, 1.0

    # Test matrix multiplication
    N1,N2 = 3,2
    a = @MMatrix rand(N1,N2)
    b = @MMatrix rand(N2,N2)
    c = @MMatrix rand(N1,N2)
    test_muladd(a, b, c, α, β)
    test_muladd(a, b, c, 1, β)
    test_muladd(a, b, c, α, 2)
    test_muladd(a, b, c, α, Float32(3))

    N1,N2 = 5,6
    a = @MMatrix rand(N1,N2)
    b = @MMatrix rand(N2,N2)
    c = @MMatrix rand(N1,N2)
    test_muladd(a, b, c, α, β)

    N1,N2 = 14,16
    a = @MMatrix rand(N1,N2)
    b = @MMatrix rand(N2,N2)
    c = @MMatrix rand(N1,N2)
    test_muladd(a, b, c, α, β)

    N1,N2 = 3,2
    a = SizedMatrix{N1,N2}(rand(N1,N2))
    b = SizedMatrix{N2,N2}(rand(N2,N2))
    c = SizedMatrix{N1,N2}(rand(N1,N2))
    test_muladd(a, b, c, α, β)

    N1,N2 = 5,6
    a = SizedMatrix{N1,N2}(rand(N1,N2))
    b = SizedMatrix{N2,N2}(rand(N2,N2))
    c = SizedMatrix{N1,N2}(rand(N1,N2))
    test_muladd(a, b, c, α, β)

    N1,N2 = 14,16
    a = SizedMatrix{N1,N2}(rand(N1,N2))
    b = SizedMatrix{N2,N2}(rand(N2,N2))
    c = SizedMatrix{N1,N2}(rand(N1,N2))
    test_muladd(a, b, c, α, β)

    # Test matrix-vector multiplication
    N = 15
    a = @MMatrix rand(N,N)
    b = @MVector rand(N)
    c = @MVector rand(N)
    test_muladd(a,b,c,α,β)

    N = 15
    a = rand(SizedMatrix{N,N})
    b = rand(SizedVector{N})
    c = rand(SizedVector{N})
    test_muladd(a,b,c,α,β)

    # Test outer products
    N = 5
    a = @MVector rand(N)
    b = @MVector rand(N)
    c = @MMatrix rand(N,N)
    test_muladd(a,b',c,α,β, false)

    N = 15
    a = rand(SizedVector{N})
    b = rand(SizedVector{N})
    c = rand(SizedMatrix{N,N})
    test_muladd(a,b',c,α,β, false)
end

#=
Summary:
All MMatrix sizes run without allocations
All MMatrix sizes have low compile times

Slow compilation:
    MMatrix initialization
    MMatrix multiplication (not in place)
=#
