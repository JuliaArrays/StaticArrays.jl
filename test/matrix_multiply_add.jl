
function test_muladd(a,b,c, α, β)
    c0 = copy(c)
    mul!(c,a,b,α,β)
    @test c ≈ a*b*α + c0*β
    c .= c0
    mul!(c,a,b)
    @test c ≈ a*b
    @test (@allocated mul!(c,a,b)) == 0
    @test (@allocated mul!(c,a,b,α,β)) == 0
end

@testset "Matrix multiply-add" begin
    α,β = 2.0, 1.0

    # Test matrix multiplication
    N = 3
    a = @MMatrix rand(N,N)
    b = @MMatrix rand(N,N)
    c = @MMatrix rand(N,N)
    test_muladd(a, b, c, α, β)
    test_muladd(a, b, c, 1, β)
    test_muladd(a, b, c, α, 2)
    test_muladd(a, b, c, α, Float32(3))

    N = 5
    a = @MMatrix rand(N,N)
    b = @MMatrix rand(N,N)
    c = @MMatrix rand(N,N)
    test_muladd(a, b, c, α, β)

    N = 15
    a = @MMatrix rand(N,N)
    b = @MMatrix rand(N,N)
    c = @MMatrix rand(N,N)
    test_muladd(a, b, c, α, β)

    N = 3
    a = SizedMatrix{N,N}(rand(N,N))
    b = SizedMatrix{N,N}(rand(N,N))
    c = SizedMatrix{N,N}(rand(N,N))
    test_muladd(a, b, c, α, β)

    N = 5
    a = SizedMatrix{N,N}(rand(N,N))
    b = SizedMatrix{N,N}(rand(N,N))
    c = SizedMatrix{N,N}(rand(N,N))
    test_muladd(a, b, c, α, β)

    N = 15
    a = SizedMatrix{N,N}(rand(N,N))
    b = SizedMatrix{N,N}(rand(N,N))
    c = SizedMatrix{N,N}(rand(N,N))
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
    test_muladd(a, b', c, α, β)

    N = 15
    a = rand(SizedVector{N})
    b = rand(SizedVector{N})
    c = rand(SizedMatrix{N,N})
end

#=
Summary:
All MMatrix sizes run without allocations
All MMatrix sizes have low compile times

Slow compilation:
    MMatrix initialization
    MMatrix multiplication (not in place)
=#
