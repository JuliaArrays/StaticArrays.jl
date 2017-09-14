@testset "qr" begin
    function test_qr(arr)
        QR = @inferred qr(arr)
        @test QR isa Tuple
        @test length(QR) == 2
        Q, R = QR
        @test Q isa StaticMatrix
        @test R isa StaticMatrix

        Q_ref,R_ref = qr(Matrix(arr))
        @test Q ≈ Q_ref
        @test R ≈ R_ref

        pivot = Val{true}
        QRp = @inferred qr(arr,pivot)
        @test QRp isa Tuple
        @test length(QRp) == 3
        Q, R, p= QRp
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
            (@SMatrix randn(10, 12)),
            map(BigFloat, @SMatrix randn(1,1)),
            (@SMatrix zeros(Int,10,10)),
            map(Complex128, @MMatrix randn(2,3))
        ]
        test_qr(arr)
    end
end
