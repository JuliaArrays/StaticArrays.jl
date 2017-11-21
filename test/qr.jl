using StaticArrays, Base.Test

Base.randn(::Type{BigFloat}) = BigFloat(randn(Float64))
Base.randn(::Type{BigFloat}, I::Integer) = [randn(BigFloat) for i=1:I]
Base.randn(::Type{Int}) = rand(-9:9)
Base.randn(::Type{Int}, I::Integer) = [randn(Int) for i=1:I]
Base.randn(::Type{Complex{T}}) where T = Complex{T}(randn(T,2)...)
Base.randn(::Type{Complex}) = randn(Complex{Float64})

srand(42)
@testset "QR decomposition" begin
    function test_qr(arr)

        # thin=true case
        QR = @inferred qr(arr)
        @test QR isa Tuple
        @test length(QR) == 2
        Q, R = QR
        @test Q isa StaticMatrix
        @test R isa StaticMatrix

        Q_ref,R_ref = qr(Matrix(arr))
        @test abs.(Q) ≈ abs.(Q_ref) # QR is unique up to diag(Q) signs
        @test abs.(R) ≈ abs.(R_ref)
        @test Q*R ≈ arr
        @test Q'*Q ≈ eye(Q'*Q)
        @test istriu(R)

        # fat (thin=false) case
        QR = @inferred qr(arr, Val(false), Val(false))
        @test QR isa Tuple
        @test length(QR) == 2
        Q, R = QR
        @test Q isa StaticMatrix
        @test R isa StaticMatrix

        Q_ref,R_ref = qr(Matrix(arr), thin=false)
        @test abs.(Q) ≈ abs.(Q_ref) # QR is unique up to diag(Q) signs
        @test abs.(R) ≈ abs.(R_ref)
        R0 = vcat(R, @SMatrix(zeros(size(arr)[1]-size(R)[1], size(R)[2])) )
        @test Q*R0 ≈ arr
        @test Q'*Q ≈ eye(Q'*Q)
        @test istriu(R)

        # pivot=true cases are not released yet
        pivot = Val(true)
        QRp = @inferred qr(arr, pivot)
        @test QRp isa Tuple
        @test length(QRp) == 3
        Q, R, p = QRp
        @test Q isa StaticMatrix
        @test R isa StaticMatrix
        @test p isa StaticVector

        Q_ref,R_ref, p_ref = qr(Matrix(arr), pivot)
        @test Q ≈ Q_ref
        @test R ≈ R_ref
        @test p == p_ref
    end

    @test_throws ArgumentError qr(@SMatrix randn(1,2); thin=false)

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
                   (@SMatrix randn(17,18)),    # fallback to LAPACK
                   (@SMatrix randn(18,17))
               ]
        test_qr(arr)
    end
end
