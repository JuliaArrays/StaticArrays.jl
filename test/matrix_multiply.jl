@testset "Matrix multiplication" begin 
    @testset "Matrix-vector" begin
        m = @SMatrix [1 2; 3 4]
        v = @SVector [1, 2]
        @test m*v === @SVector [5, 11]
        # More complicated eltype inference
        v2 = @SVector [CartesianIndex((1,3)), CartesianIndex((3,1))]
        x = @inferred(m*v2)
        @test isa(x, SVector{2,CartesianIndex{2}})
        @test x == @SVector [CartesianIndex((7,5)), CartesianIndex((15,13))]

        v3 = [1, 2]
        @test m*v3 === @SVector [5, 11]

        m2 = @MMatrix [1 2; 3 4]
        v4 = @MVector [1, 2]
        @test (m2*v4)::MVector == @MVector [5, 11]

        m3 = @SArray [1 2; 3 4]
        v5 = @SArray [1, 2]
        @test m3*v5 === @SArray [5, 11]

        m4 = @MArray [1 2; 3 4]
        v6 = @MArray [1, 2]
        @test (m4*v6)::MArray == @MArray [5, 11]

        m5 = @SMatrix [1.0 2.0; 3.0 4.0]
        v7 = [1.0, 2.0]
        @test (m5*v7)::SVector ≈ @SVector [5.0, 11.0]

        m6 = @SMatrix Float32[1.0 2.0; 3.0 4.0]
        v8 = Float64[1.0, 2.0]
        @test (m6*v8)::SVector{2,Float64} ≈ @SVector [5.0, 11.0]

    end

    @testset "Vector-matrix" begin
        m = @SMatrix [1 2 3 4]
        v = @SVector [1, 2]
        @test v*m === @SMatrix [1 2 3 4; 2 4 6 8]
    end

    @testset "Matrix-matrix" begin
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]
        nc = @SMatrix [CartesianIndex((2,2)) CartesianIndex((3,3));
                       CartesianIndex((4,4)) CartesianIndex((5,5))]
        @test m*n  === @SMatrix [10 13; 22 29]
        @test m*nc === @SMatrix [CartesianIndex((10,10)) CartesianIndex((13,13));
                                 CartesianIndex((22,22)) CartesianIndex((29,29))]

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

        m_array2 = rand(1:10, 16, 16) # see JuliaLang/julia#18794 for reason for variable name changes
        n_array2 = rand(1:10, 16, 16)
        a_array2 = m_array2*n_array2

        m2 = SMatrix{16,16}(m_array2)
        n2 = SMatrix{16,16}(n_array2)
        @test m2*n2 === SMatrix{16,16}(a_array2)

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
        v = @SVector [2, 4]
        v2 = [2, 4]
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]

        outvec = MVector{2,Int}()
        A_mul_B!(outvec, m, v)
        @test outvec == @MVector [10,22]
        outvec2 = Vector{Int}(2)
        A_mul_B!(outvec2, m, v2)
        @test outvec2 == [10,22]

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
        m_array_2 = rand(1:10, 10, 10)
        n_array_2 = rand(1:10, 10, 10)
        a_array_2 = m_array_2*n_array_2

        m_2 = MMatrix{10,10}(m_array_2)
        n_2 = MMatrix{10,10}(n_array_2)
        a_2 = MMatrix{10,10,Int}()
        A_mul_B!(a_2, m_2, n_2)
        @test a_2 == a_array_2

        # BLAS used for n > 14
        m_array_3 = randn(4, 4)
        n_array_3 = randn(4, 4)
        a_array_3 = m_array_3*n_array_3

        m_3 = MMatrix{4,4}(m_array_3)
        n_3 = MMatrix{4,4}(n_array_3)
        a_3 = MMatrix{4,4,Float64}()
        A_mul_B!(a_3, m_3, n_3)
        @test a_3 ≈ a_array_3

        m_array_4 = randn(10, 10)
        n_array_4 = randn(10, 10)
        a_array_4 = m_array_4*n_array_4

        m_4 = MMatrix{10,10}(m_array_4)
        n_4 = MMatrix{10,10}(n_array_4)
        a_4 = MMatrix{10,10,Float64}()
        A_mul_B!(a_4, m_4, n_4)
        @test a_4 ≈ a_array_4

        m_array_5 = rand(1:10, 16, 16)
        n_array_5 = rand(1:10, 16, 16)
        a_array_5 = m_array_5*n_array_5

        m_5 = MMatrix{16,16}(m_array_5)
        n_5 = MMatrix{16,16}(n_array_5)
        a_5 = MMatrix{16,16,Int}()
        A_mul_B!(a_5, m_5, n_5)
        @test a_5 ≈ a_array_5

        # Float64
        vf = @SVector [2.0, 4.0]
        vf2 = [2.0, 4.0]
        mf = @SMatrix [1.0 2.0; 3.0 4.0]

        outvecf = MVector{2,Float64}()
        A_mul_B!(outvecf, mf, vf)
        @test outvecf ≈ @MVector [10.0, 22.0]
        outvec2f = Vector{Float64}(2)
        A_mul_B!(outvec2f, mf, vf2)
        @test outvec2f ≈ [10.0, 22.0]
    end
end
