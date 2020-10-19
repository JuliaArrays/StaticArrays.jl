using StaticArrays, Test, LinearAlgebra

@testset "LU utils" begin
    F = lu(SA[1 2; 3 4])

    @test @inferred((F->F.p)(F)) === SA[2, 1]
    @test @inferred((F->F.P)(F)) === SA[0 1; 1 0]

    @test occursin(r"^StaticArrays.LU.*L factor.*U factor"s, sprint(show, MIME("text/plain"), F))
end

@testset "LU decomposition ($m×$n, pivot=$pivot)" for pivot in (true, false), m in [0:4..., 15], n in [0:4..., 15]
    a = SMatrix{m,n,Int}(1:(m*n))
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
