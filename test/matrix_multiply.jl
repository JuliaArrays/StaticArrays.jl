@testset "Matrix multiplication" begin
    @testset "Matrix-vector" begin
        m = @SMatrix [1 2; 3 4]
        v = @SVector [1, 2]
        @test m*v === @SVector [5, 11]

        m = @MMatrix [1 2; 3 4]
        v = @MVector [1, 2]
        @test (m*v)::MVector == @MVector [5, 11]

        m = @SArray [1 2; 3 4]
        v = @SArray [1, 2]
        @test m*v === @SArray [5, 11]

        m = @MArray [1 2; 3 4]
        v = @MArray [1, 2]
        @test (m*v)::MArray == @MArray [5, 11]
    end

    @testset "Vector-matrix" begin
        m = @SMatrix [1 2 3 4]
        v = @SVector [1, 2]
        @test v*m === @SMatrix [1 2 3 4; 2 4 6 8]
    end

    @testset "Matrix-matrix" begin
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]
        @test m*n === @SMatrix [10 13; 22 29]

        @test m'*n === @SMatrix [14 18; 20 26]
        @test m*n' === @SMatrix [8 14; 18 32]
        @test m'*n' === @SMatrix [11 19; 16 28]
        @test m.'*n === @SMatrix [14 18; 20 26]
        @test m*n.' === @SMatrix [8 14; 18 32]
        @test m.'*n.' === @SMatrix [11 19; 16 28]

        m = @MMatrix [1 2; 3 4]
        n = @MMatrix [2 3; 4 5]
        @test (m*n)::MMatrix == @MMatrix [10 13; 22 29]

        m = @SArray [1 2; 3 4]
        n = @SArray [2 3; 4 5]
        @test m*n === @SArray [10 13; 22 29]

        m = @MArray [1 2; 3 4]
        n = @MArray [2 3; 4 5]
        @test (m*n)::MArray == @MArray [10 13; 22 29]

        # Alternative methods used between 8 < n <= 14 and n > 14
        m_array = rand(1:10, 10, 10)
        n_array = rand(1:10, 10, 10)
        a_array = m_array*n_array

        m = SMatrix{10,10}(m_array)
        n = SMatrix{10,10}(n_array)
        @test m*n === SMatrix{10,10}(a_array)

        m_array = rand(1:10, 16, 16)
        n_array = rand(1:10, 16, 16)
        a_array = m_array*n_array

        m = SMatrix{16,16}(m_array)
        n = SMatrix{16,16}(n_array)
        @test m*n === SMatrix{16,16}(a_array)

        # Mutating types follow different behaviour
        m_array = rand(1:10, 10, 10)
        n_array = rand(1:10, 10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        @test (m*n)::MMatrix == a_array

        m_array = rand(1:10, 16, 16)
        n_array = rand(1:10, 16, 16)
        a_array = m_array*n_array

        m = MMatrix{16,16}(m_array)
        n = MMatrix{16,16}(n_array)
        @test (m*n)::MMatrix == a_array

        # Mutating BLAS types follow yet different behaviour
        m_array = randn(4, 4)
        n_array = randn(4, 4)
        a_array = m_array*n_array

        m = MMatrix{4,4}(m_array)
        n = MMatrix{4,4}(n_array)
        @test m*n::MMatrix ≈ a_array

        m_array = randn(10, 10)
        n_array = randn(10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        @test m*n::MMatrix ≈ a_array

        m_array = randn(16, 16)
        n_array = randn(16, 16)
        a_array = m_array*n_array

        m = MMatrix{16,16}(m_array)
        n = MMatrix{16,16}(n_array)
        @test m*n::MMatrix ≈ a_array
    end

    @testset "A_mul_B!" begin
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]

        a = MMatrix{2,2,Int,4}()
        A_mul_B!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [10 13; 22 29]

        Ac_mul_B!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [14 18; 20 26]
        A_mul_Bc!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [8 14; 18 32]
        Ac_mul_Bc!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [11 19; 16 28]
        At_mul_B!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [14 18; 20 26]
        A_mul_Bt!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [8 14; 18 32]
        At_mul_Bt!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [11 19; 16 28]

        a2 = MArray{(2,2),Int,2,4}()
        A_mul_B!(a2, m, n)
        @test a2::MArray{(2,2),Int,2,4} == @MArray [10 13; 22 29]

        # Alternative builtin method used for n > 8
        m_array = rand(1:10, 10, 10)
        n_array = rand(1:10, 10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        a = MMatrix{10,10,Int}()
        A_mul_B!(a, m, n)
        @test a == a_array

        # BLAS used for n > 14
        m_array = randn(4, 4)
        n_array = randn(4, 4)
        a_array = m_array*n_array

        m = MMatrix{4,4}(m_array)
        n = MMatrix{4,4}(n_array)
        a = MMatrix{4,4,Float64}()
        A_mul_B!(a, m, n)
        @test a ≈ a_array

        m_array = randn(10, 10)
        n_array = randn(10, 10)
        a_array = m_array*n_array

        m = MMatrix{10,10}(m_array)
        n = MMatrix{10,10}(n_array)
        a = MMatrix{10,10,Float64}()
        A_mul_B!(a, m, n)
        @test a ≈ a_array

        m_array = rand(1:10, 16, 16)
        n_array = rand(1:10, 16, 16)
        a_array = m_array*n_array

        m = MMatrix{16,16}(m_array)
        n = MMatrix{16,16}(n_array)
        a = MMatrix{16,16,Int}()
        A_mul_B!(a, m, n)
        @test a ≈ a_array
    end
end
