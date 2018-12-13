using StaticArrays, Test, LinearAlgebra, Random

Base.randn(::Type{BigFloat}) = BigFloat(randn(Float64))
Base.randn(::Type{BigFloat}, I::Integer) = [randn(BigFloat) for i=1:I]
Base.randn(::Type{Int}) = rand(-9:9)
Base.randn(::Type{Int}, I::Integer) = [randn(Int) for i=1:I]
Base.randn(::Type{Complex{T}}) where T = Complex{T}(randn(T,2)...)
Base.randn(::Type{Complex}) = randn(Complex{Float64})

Random.seed!(42)
@testset "QR decomposition" begin
    function test_qr(arr)

        T = eltype(arr)

        QR = @inferred qr(arr)
        @test QR isa StaticArrays.QR
        Q, R = QR # destructing via iteration
        @test Q isa StaticMatrix
        @test R isa StaticMatrix
        @test eltype(Q) == eltype(R) == typeof((one(T)*zero(T) + zero(T))/norm([one(T)]))

        Q_ref, R_ref = qr(Matrix(arr))
        @test abs.(Q) ≈ abs.(Matrix(Q_ref)) # QR is unique up to diag(Q) signs
        @test abs.(R) ≈ abs.(R_ref)
        @test Q*R ≈ arr
        @test Q'*Q ≈ one(Q'*Q)
        @test istriu(R)

        # pivot=true cases has no StaticArrays specific version yet
        # but fallbacks to LAPACK
        pivot = Val(true)
        QRp = @inferred qr(arr, pivot)
        @test QRp isa StaticArrays.QR
        Q, R, p = QRp
        @test Q isa StaticMatrix
        @test R isa StaticMatrix
        @test p isa StaticVector

        Q_ref, R_ref, p_ref = qr(Matrix(arr), pivot)
        @test Q ≈ Matrix(Q_ref)
        @test R ≈ R_ref
        @test p == p_ref
    end

    for eltya in (Float32, Float64, BigFloat, Int),
            rel in (real, complex),
                sz in [(3,3), (3,4), (4,3)]
        arr = SMatrix{sz[1], sz[2], rel(eltya), sz[1]*sz[2]}( [randn(rel(eltya)) for i = 1:sz[1], j = 1:sz[2]] )
        test_qr(arr)
    end
    # some special cases
    for arr in [
                   (@MMatrix randn(3,2)),
                   (@MMatrix randn(2,3)),
                   (@SMatrix([0 1 2; 0 2 3; 0 3 4; 0 4 5])),
                   (@SMatrix zeros(Int,4,4)),
                   (@SMatrix([1//2 1//1])),
                   (@SMatrix randn(17,18)),    # fallback to LAPACK
                   (@SMatrix randn(18,17))
               ]
        test_qr(arr)
    end
end
