@testset "QR decomposition" begin
    function test_qr(arr)
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
        @test abs.(Q_ref'*Q) ≈ eye(Q_ref'*Q)
        @test Q_ref'*Q * R ≈ R_ref

        pivot = Val{true}
        QRp = @inferred qr(arr,pivot)
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

    for arr in [
            (@MMatrix randn(2,2)),
            [(@SMatrix randn(i, j)) for i=3:3 for j=max(i-1,1):i+1]...,
            (@SMatrix randn(10, 12)),
            (@SMatrix([0 1 2; 0 2 3; 0 3 4; 0 4 5])),
            (@SMatrix zeros(Int,5,5)),
            map(Complex128, @SMatrix randn(2,3)),
            map(Complex128, @SMatrix randn(3,2)),
            map(BigFloat, @SMatrix randn(1,1)),
            map(Complex{BigFloat}, @MMatrix randn(2,3)),
            map(Complex{BigFloat}, @SMatrix randn(3,2)),
            (@SMatrix randn(19, 2)),
            (@SMatrix randn(2, 19))
        ]
        test_qr(arr)
    end
end
