using StaticArrays, Test, LinearAlgebra

@testset "LU" begin

@testset "LU utils" begin
    F = lu(SA[1 2; 3 4])

    @test @inferred((F->F.p)(F)) === SA[2, 1]
    @test @inferred((F->F.P)(F)) === SA[0 1; 1 0]

    @test occursin(r"^StaticArrays.LU.*L factor.*U factor"s, sprint(show, MIME("text/plain"), F))
end

@testset "LU decomposition ($m×$n, pivot=$pivot, wrapper=$wrapper)" for pivot in (true, false), m in [0:4..., 15], n in [0:4..., 15], wrapper in [identity, Symmetric, Hermitian]

    a = if m == n && m > 0
        wrapper(SMatrix{m,n,Int}(1:(m*n)))
    elseif wrapper !== identity
        continue
    else
        SMatrix{m,n,Int}(1:(m*n))
    end
    l, u, p = @inferred(lu(a, Val{pivot}(); check = false))

    # expected types
    @test p isa SVector{m,Int}
    if m==n
        @test l isa LowerTriangular{<:Any,<:SMatrix{m,n}}
        @test u isa UpperTriangular{<:Any,<:SMatrix{m,n}}
    else
        @test l isa SMatrix{m,min(m,n)}
        @test u isa SMatrix{min(m,n),n}
    end

    if pivot
        # p is a permutation
        @test sort(p) == collect(1:m)
    else
        @test p == collect(1:m)
    end

    # l is unit lower triangular
    for i=1:m, j=(i+1):size(l,2)
        @test iszero(l[i,j])
    end
    for i=1:size(l,2)
        @test l[i,i] == 1
    end

    # u is upper triangular
    for i=1:size(u,1), j=1:i-1
        @test iszero(u[i,j])
    end

    # decomposition is correct
    l_u = l*u
    @test l*u ≈ a[p,:]

end

@testset "LU division ($m×$n)" for m in [1:4..., 15], n in [1:4..., 15]
    a = SMatrix{m,m,Float64}(rand(Float64,m,m))
    a_lu = lu(a)
    b_col = SMatrix{m,n,Float64}(rand(Float64,m,n))
    b_line = SMatrix{n,m,Float64}(rand(Float64,n,m))

    # test if / and \ work with lu:
    @test a\b_col ≈ a_lu\b_col
    @test b_line/a ≈ b_line/a_lu
end

@testset "LU singularity check" for m in [2, 3, 20], n in [2, 3, 20]
    # NOTE: large dimensions test fallback to LinearAlgebra.lu
    A = ones(SMatrix{m,n})
    @test_throws SingularException lu(A)
    @test !issuccess(lu(A; check = false))
end

@testset "LU rational zero pivots" for pivot in (true, false), constructor in (SMatrix, MMatrix)
    # Cover both the single-column and recursive static implementations,
    # including a zero reached only after an elimination step.
    for values in (zeros(Rational{Int}, 3, 1), zeros(Rational{Int}, 3, 2),
                   zeros(Rational{Int}, 2, 3), ones(Rational{Int}, 3, 3))
        A = constructor{size(values, 1), size(values, 2)}(values)
        F = @inferred lu(A, Val(pivot); check=false)
        @test eltype(F.L) == Rational{Int}
        @test eltype(F.U) == Rational{Int}
        @test F.L * F.U == A[F.p, :]
        @test !issuccess(F)
        @test_throws SingularException lu(A, Val(pivot))
    end
    # Original determinant report: the zero fourth pivot is encountered by
    # the recursive LU route for a 5x5 static matrix.
    for n in 2:5
        values = Matrix{Rational{Int}}(I, n, n)
        values[n-1, n-1] = 0
        A = constructor{n, n}(values)
        @test @inferred(det(A)) == 0//1
    end
    A = constructor{3, 3}([1//1 2//1 0//1; 0//1 3//1 4//1; 5//1 0//1 6//1])
    F = @inferred lu(A, Val(pivot))
    @test F.L * F.U == A[F.p, :]
    @test issuccess(F)
    @test eltype(F.L) == Rational{Int}
    @test det(A) == 58//1
end

@testset "LU nonfinite reciprocal multipliers" for T in (Float32, Float64, ComplexF64)
    # Preserve the existing zero-multiplier fallback, including nonzero
    # subnormal pivots whose reciprocal overflows and NaN pivots.
    for pivot in (zero(T), T(NaN), T(Inf), T(nextfloat(zero(typeof(real(zero(T)))))))
        for A in (SMatrix{3, 1}(pivot, one(T), one(T)),
                  SMatrix{3, 2}(pivot, one(T), one(T), one(T), one(T), one(T)))
            F = @inferred lu(A, Val(false); check=false)
            @test eltype(F.L) == T
            @test all(iszero, F.L[2:end, 1])
            @test isequal(F.U[1, 1], pivot)
        end
    end
end

@testset "LU method ambiguity" begin
    # Issue #920; just test that methods do not throw an ambiguity error when called
    for A in ((@SMatrix [1.0 2.0; 3.0 4.0]), (@SMatrix [1.0 2.0 3.0; 4.0 5.0 6.0]))
        @test isa(lu(A),              StaticArrays.LU)
        @test isa(lu(A, Val(true)),   StaticArrays.LU)
        @test isa(lu(A, Val(false)),  StaticArrays.LU)
        @test isa(lu(A; check=false), StaticArrays.LU)
        @test isa(lu(A; check=true),  StaticArrays.LU)
    end
end

if isdefined(LinearAlgebra, :PivotingStrategy)
    for N = (3, 15)
        A = (@SMatrix randn(N,N))
        @test lu(A, Val(false)) == lu(A, NoPivot())
        @test lu(A, Val(true)) == lu(A, RowMaximum())
    end
end

end # @testset "LU"
