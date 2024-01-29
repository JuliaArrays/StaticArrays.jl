using StaticArrays, Test, LinearAlgebra

@testset "AbstractArray interface" begin
    @testset "size and length" begin
        m = @SMatrix [1 2 3; 4 5 6; 7 8 9; 10 11 12]

        @test length(m) == 12
        @test IndexStyle(m) == IndexLinear()
        @test Base.isassigned(m, 2, 2) == true
        @test eachindex(m) isa SOneTo
    end

    @testset "axes" begin
        v = @SVector [1, 2, 3]
        @test @inferred(axes(v)) == (SOneTo(3),)
        for T in (Adjoint, Transpose)
            @test @inferred(axes(T(v))) == (SOneTo(1), SOneTo(3))
        end

        m = @SMatrix [1 2; 3 4]
        @test @inferred(axes(m)) == (SOneTo(2), SOneTo(2))
        for T in (Adjoint, Transpose, Diagonal, Symmetric, Hermitian, UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
            @test @inferred(axes(T(m))) == (SOneTo(2), SOneTo(2))
        end
    end

    @testset "strides" begin
        m1 = MArray{Tuple{3, 4, 5}}(rand(Int, 3, 4, 5))
        m2 = SizedArray{Tuple{3,4,5}}(rand(Int, 3, 4, 5))
        @test strides(m1) === (1, 3, 12)
        @test strides(m2) === (1, 3, 12)
    end

    @testset "similar_type" begin
        @test @inferred(similar_type(SVector{3,Int})) == SVector{3,Int}
        @test @inferred(similar_type(@SVector [1,2,3])) == SVector{3,Int}

        @test @inferred(similar_type(SVector{3,Int}, Float64)) == SVector{3,Float64}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Size(2))) == SVector{2, Int}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64, Size(2))) == SVector{2, Float64}
        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64, Size(2))) == SVector{2, Float64}

        @test @inferred(similar_type(SMatrix{3,3,Int,9}, Float64)) == SMatrix{3, 3, Float64, 9}
        @test @inferred(similar_type(SVector{2,Int}, Size(3,3))) == SMatrix{3, 3, Int, 9}
        @test @inferred(similar_type(SVector{2,Int}, Float64, Size(3,3))) == SMatrix{3, 3, Float64, 9}

        @test @inferred(similar_type(SArray{Tuple{4,4,4},Int,3,64}, Float64)) == SArray{Tuple{4,4,4}, Float64, 3, 64}
        @test @inferred(similar_type(SVector{2,Int}, Size(3,3,3))) == SArray{Tuple{3,3,3}, Int, 3, 27}
        @test @inferred(similar_type(SVector{2,Int}, Float64, Size(3,3,3))) == SArray{Tuple{3,3,3}, Float64, 3, 27}

        # Some specializations for the mutable case
        @test @inferred(similar_type(MVector{3,Int}, Float64)) == MVector{3,Float64}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Size(2))) == MVector{2, Int}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64, Size(2))) == MVector{2, Float64}
        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64, Size(2))) == MVector{2, Float64}

        @test @inferred(similar_type(MMatrix{3,3,Int,9}, Float64)) == MMatrix{3, 3, Float64, 9}
        @test @inferred(similar_type(MVector{2,Int}, Size(3,3))) == MMatrix{3, 3, Int, 9}
        @test @inferred(similar_type(MVector{2,Int}, Float64, Size(3,3))) == MMatrix{3, 3, Float64, 9}

        @test @inferred(similar_type(MArray{Tuple{4,4,4},Int,3,64}, Float64)) == MArray{Tuple{4,4,4}, Float64, 3, 64}
        @test @inferred(similar_type(MVector{2,Int}, Size(3,3,3))) == MArray{Tuple{3,3,3}, Int, 3, 27}
        @test @inferred(similar_type(MVector{2,Int}, Float64, Size(3,3,3))) == MArray{Tuple{3,3,3}, Float64, 3, 27}

        # wrapped mutable cases (issue #828)
        for Wrapper in [Symmetric, Hermitian, Adjoint, Transpose, UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular]
            @test @inferred(similar_type(Wrapper{Int,MArray{Tuple{4,4},Int,2,16}}, Float64)) === MArray{Tuple{4,4}, Float64, 2, 16}
        end
        @test @inferred(similar_type(Diagonal{Int,MArray{Tuple{4},Int,1,4}}, Float64)) === MArray{Tuple{4,4}, Float64, 2, 16}
    end

    @testset "similar" begin
        sv = @SVector [1,2,3]
        sm = @SMatrix [1 2; 3 4]
        sa = SArray{Tuple{1,1,1},Int,3,1}((1,))
        sn = @SVector [1, missing]

        @test isa(@inferred(similar(sv)), MVector{3,Int})
        @test isa(@inferred(similar(sv, Float64)), MVector{3,Float64})
        @test isa(@inferred(similar(sv, Size(4))), MVector{4,Int})
        @test isa(@inferred(similar(sv, Float64, Size(4))), MVector{4,Float64})

        @test isa(@inferred(similar(sm)), MMatrix{2,2,Int,4})
        @test isa(@inferred(similar(sm, Float64)), MMatrix{2,2,Float64,4})
        @test isa(@inferred(similar(sv, Size(3,3))), MMatrix{3,3,Int,9})
        @test isa(@inferred(similar(sv, Float64, Size(3,3))), MMatrix{3,3,Float64,9})
        @test isa(@inferred(similar(sn)), SizedVector{2, Union{Missing, Int}})
        @test isa(@inferred(similar(sn, Float64, Size(3, 3))), MMatrix{3, 3, Float64, 9})

        @test isa(@inferred(similar(sa)), MArray{Tuple{1,1,1},Int,3,1})
        @test isa(@inferred(similar(SArray{Tuple{1,1,1},Int,3,1})), MArray{Tuple{1,1,1},Int,3,1})
        @test isa(@inferred(similar(sa, Float64)), MArray{Tuple{1,1,1},Float64,3,1})
        @test isa(@inferred(similar(SArray{Tuple{1,1,1},Int,3,1}, Float64)), MArray{Tuple{1,1,1},Float64,3,1})
        @test isa(@inferred(similar(sv, Size(3,3,3))), MArray{Tuple{3,3,3},Int,3,27})
        @test isa(@inferred(similar(sv, Float64, Size(3,3,3))), MArray{Tuple{3,3,3},Float64,3,27})

        @test isa(@inferred(similar(Diagonal{Int}, Size(2,2))), MArray{Tuple{2, 2}, Int, 2, 4})
        @test isa(@inferred(similar(SizedArray, Int, Size(2,2))), SizedArray{Tuple{2, 2}, Int, 2, 2})
        @test isa(@inferred(similar(Matrix{Int}, Int, Size(2,2))), SizedArray{Tuple{2, 2}, Int, 2, 2})

        @testset "disambiguate similar" begin
            struct CustomArray{T} <: AbstractVector{T}
                sz :: Int
            end

            Base.size(C::CustomArray) = (C.sz,)
            Base.getindex(C::CustomArray{T}, i::Int) where {T} = T(i)
            Base.similar(C::CustomArray, ::Type{T}, ax::Tuple{Vararg{Int}}) where {T} =
                Array{T}(undef, ax)
            function Base.similar(C::CustomArray, ::Type{T}, ax::Tuple{Vararg{Union{Int, SOneTo, Base.OneTo{Int}}}}) where {T}
                sz = last.(ax)
                Array{T}(undef, sz)
            end

            c = CustomArray{Int}(4)
            for (ax, sz) in (((SOneTo(2), Base.OneTo(3)), (2,3)),
                                ((2, SOneTo(2), Base.OneTo(3)), (2,2,3)))
                for A in (similar(c, Float64, ax), similar(c, Float64, ax...))
                    @test A isa Array{Float64, length(sz)}
                    @test size(A) == sz
                end
            end

            @test similar(c, ()) isa Array{Int,0}
            @test similar(c, Float64, ()) isa Array{Float64,0}

            @test size(similar(zeros(), (1,1,1,SOneTo(1)))) == (1,1,1,1)

            # ensure that the more specific Base method works
            @test similar(1:2, ()) isa AbstractArray{Int,0}
        end
    end

    @testset "similar and Base.Slice/IdentityUnitRange (issues #548, #556)" begin
        v = @SVector [1,2,3]
        m = @SMatrix [1 2 3; 4 5 6]
        @test similar(v, Int, SOneTo(3)) isa MVector{3,Int}
        @test similar(v, Int, SOneTo(3), SOneTo(4)) isa MMatrix{3,4,Int}
        @test similar(v, Int, 3, SOneTo(4)) isa Matrix
        @test similar(v, SOneTo(3)) isa MVector{3,Int}
        @test similar(v, SOneTo(3), SOneTo(4)) isa MMatrix{3,4,Int}
        @test similar(v, 3, SOneTo(4)) isa Matrix

        @test m[:, 1:2] isa Matrix
        @test m[:, [true, false, false]] isa Matrix
        @test m[:, SOneTo(2)] isa SMatrix{2, 2, Int}
        @test m[:, :] isa SMatrix{2, 3, Int}
        @test m[:, 1] isa SVector{2, Int}
        @test m[2, :] isa SVector{3, Int}

        # Test case that failed in AstroLib.jl
        r = @view(m[:, 2:3]) * @view(v[1:2])
        @test r == m[:, 2:3] * v[1:2] == Array(m)[:, 2:3] * Array(v)[1:2]
    end


    @testset "reshape" begin
        @test @inferred(reshape(SVector(1,2,3,4), axes(SMatrix{2,2}(1,2,3,4)))) === SMatrix{2,2}(1,2,3,4)
        @test @inferred(reshape(SVector(1,2,3,4), Size(2,2))) === SMatrix{2,2}(1,2,3,4)
        @test @inferred(reshape([1,2,3,4], Size(2,2)))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test @inferred(reshape([1,2,3,4], axes(SMatrix{2,2}(1,2,3,4))))::SizedArray{Tuple{2,2},Int,2,1} == [1 3; 2 4]
        @test @inferred(reshape([1 2 3 4], axes(SMatrix{2,2}(1,2,3,4))))::SizedArray{Tuple{2,2},Int,2,2} == [1 3; 2 4]
        @test_throws DimensionMismatch reshape([1 2; 3 4], Size(2,2,2))
        @test_throws DimensionMismatch reshape([1 2 3], axes(SMatrix{2,2}(1,2,3,4)))

        @test @inferred(vec(SMatrix{2, 2}([1 2; 3 4])))::SVector{4,Int} == [1, 3, 2, 4]
        a = @MVector [1, 2, 3, 4]
        @test @inferred(vec(a)) === a

        as = SizedVector(a)
        @test @inferred(vec(as)) === as

        # AbstractArray
        # CartesianIndex
        @test reshape(view(ones(4, 4), 1:4, 1:2), Size(4, 2)) == SMatrix{4,2}(ones(4, 2))
        # IndexLinear
        @test reshape(view(ones(4, 4), 1, 1:4), Size(4, 1)) == SMatrix{4,1}(ones(4, 1))
        @test_throws DimensionMismatch reshape(view(ones(4,4), 1:4, 1:2), Size(5, 2))

        # mutation
        m = @MMatrix [1 2; 3 4]
        mr = reshape(m, SOneTo(4))
        mr[2] = 10
        @test m == SA[1 2; 10 4]

        mrs = reshape(m, Size(4))
        mrs[2] = 10
        @test m == SA[1 2; 10 4]

        ms = SizedMatrix{2,2}([1 2; 3 4])
        msr = reshape(ms, SOneTo(4))
        msr[2] = 10
        @test ms == SA[1 2; 10 4]
    end

    @testset "copy" begin
        M = [1 2; 3 4]
        SM = SMatrix{2, 2}(M)
        MM = MMatrix{2, 2}(M)
        SizeM = SizedMatrix{2,2}(M)
        @test @inferred(copy(SM)) === @SMatrix [1 2; 3 4]
        @test @inferred(copy(MM))::MMatrix == M
        @test copy(SM).data !== M
        @test copy(SizeM).data !== M
    end

    @testset "reverse" begin
        @test @inferred(reverse(SVector(1, 2, 3))) ≡ SVector(3, 2, 1)
        m = MVector(1, 2, 3)
        @test @inferred(reverse(m))::typeof(m) == MVector(3, 2, 1)
    end

    @testset "rotate" begin
        M = [1 2; 3 4]
        SM = SMatrix{2, 2}(M)
        @test @inferred(rotl90(SM)) === @SMatrix [2 4; 1 3]
        @test @inferred(rot180(SM)) === @SMatrix [4 3; 2 1]
        @test @inferred(rotr90(SM)) === @SMatrix [3 1; 4 2]
        M23 = rand(2, 3)
        SM23 = SMatrix{2, 3}(M23)
        @test @inferred(rotl90(SM23)) == rotl90(M23)
        @test @inferred(rot180(SM23)) == rot180(M23)
        @test @inferred(rotr90(SM23)) == rotr90(M23)
    end

    @testset "Conversion to AbstractArray" begin
        # Issue #746
        # conversion to AbstractArray changes the eltype from Int to Float64
        sv = SVector(1,2)
        @test @inferred(convert(AbstractArray{Float64}, sv)) isa SVector{2,Float64}
        @test @inferred(convert(AbstractVector{Float64}, sv)) isa SVector{2,Float64}
        @test convert(AbstractArray{Float64}, sv) == sv
        @test convert(AbstractArray{Int}, sv) === sv
        sm = SMatrix{2,2}(1,2,3,4)
        @test @inferred(convert(AbstractArray{Float64,2}, sm)) isa SMatrix{2,2,Float64}
        @test convert(AbstractArray{Float64,2}, sm) == sm
        @test convert(AbstractArray{Int,2}, sm) === sm
        mv = MVector(1, 2, 3)
        @test @inferred(convert(AbstractArray{Float64}, mv)) isa MVector{3,Float64}
        @test @inferred(convert(AbstractVector{Float64}, mv)) isa MVector{3,Float64}
        @test convert(AbstractArray{Float64}, mv) == mv
        @test convert(AbstractArray{Int}, mv) === mv
        mm = MMatrix{2, 2}(1, 2, 3, 4)
        @test @inferred(convert(AbstractArray{Float64,2}, mm)) isa MMatrix{2,2,Float64}
        @test convert(AbstractArray{Float64,2}, mm) == mm
        @test convert(AbstractArray{Int,2}, mm) === mm

        # Test some of the types in StaticMatrixLike
        sym = Symmetric(SA[1 2; 2 3])
        @test @inferred(convert(AbstractArray{Float64}, sym)) isa Symmetric{Float64,SMatrix{2,2,Float64,4}}
        @test @inferred(convert(AbstractArray{Float64,2}, sym)) isa Symmetric{Float64,SMatrix{2,2,Float64,4}}
        @test convert(AbstractArray{Float64}, sym) == sym
        her = Hermitian(SA[1 2+im; 2-im 3])
        @test @inferred(convert(AbstractArray{ComplexF64}, her)) isa Hermitian{ComplexF64,SMatrix{2,2,ComplexF64,4}}
        @test convert(AbstractArray{ComplexF64}, her) == her
        diag = Diagonal(SVector(1,2))
        @test @inferred(convert(AbstractArray{Float64}, diag)) isa Diagonal{Float64,SVector{2,Float64}}
        @test convert(AbstractArray{Float64}, diag) == diag
        # The following cases currently convert the SMatrix into an MMatrix, because
        # the constructor in Base invokes `similar`, rather than `convert`, on the static
        # array. This was fixed in https://github.com/JuliaLang/julia/pull/40831; so should
        # work from Julia v1.8.0-DEV.55
        trans = Transpose(SVector(1,2))
        @test_was_once_broken v"1.8.0-DEV.55" @inferred(convert(AbstractArray{Float64}, trans)) isa Transpose{Float64,SVector{2,Float64}}
        adj = Adjoint(SVector(1,2))
        @test_was_once_broken v"1.8.0-DEV.55" @inferred(convert(AbstractArray{Float64}, adj)) isa Adjoint{Float64,SVector{2,Float64}}
        uptri = UpperTriangular(SA[1 2; 0 3])
        @test_was_once_broken v"1.8.0-DEV.55" @inferred(convert(AbstractArray{Float64}, uptri)) isa UpperTriangular{Float64,SMatrix{2,2,Float64,4}}
        lotri = LowerTriangular(SA[1 0; 2 3])
        @test_was_once_broken v"1.8.0-DEV.55" @inferred(convert(AbstractArray{Float64}, lotri)) isa LowerTriangular{Float64,SMatrix{2,2,Float64,4}}
        unituptri = UnitUpperTriangular(SA[1 2; 0 1])
        @test_was_once_broken v"1.8.0-DEV.55" @inferred(convert(AbstractArray{Float64}, unituptri)) isa UnitUpperTriangular{Float64,SMatrix{2,2,Float64,4}}
        unitlotri = UnitLowerTriangular(SA[1 0; 2 1])
        @test_was_once_broken v"1.8.0-DEV.55" @inferred(convert(AbstractArray{Float64}, unitlotri)) isa UnitLowerTriangular{Float64,SMatrix{2,2,Float64,4}}
    end

    @testset "type inference in length" begin
        s1 = SA[1,2];
        s2 = SA[1,2,3];
        v = [s1, s2];
        f(v, i) = length(v[i]);
        for i in 1:2
            @test (@inferred f(v, i)) == length(v[i])
        end
    end

    @testset "reduced_indices" begin
        s = SArray{Tuple{2,2,2},Int,3,8}((1,2,3,4,5,6,7,8))
        a = Array(s)
        for i in 1:ndims(s)
            rs = @inferred Base.reduced_indices(axes(s), i)
            @test rs == Base.reduced_indices(axes(a), i)
        end
    end
end

@testset "permutedims" begin
    # vector -> one-row matrix
    @test @inferred(permutedims(SVector(1,2,3))) === SMatrix{1,3}(1,2,3)
    @test @inferred(permutedims(MVector(1,2,3))) isa MMatrix{1,3}
    @test @inferred(permutedims(MVector(1,2,3))) == [1 2 3]
    @test @inferred(permutedims(SizedVector{3}([1,2,3]))) isa SizedMatrix{1,3}
    @test @inferred(permutedims(SizedVector{3}([1,2,3]))) == [1 2 3]

    # matrix
    @test @inferred(permutedims(SMatrix{2,2}(1,2,3,4))) === SMatrix{2,2}(1,3,2,4)
    A = rand(2,3)
    @test @inferred(permutedims(SMatrix{2,3}(A))) === SMatrix{3,2}(A')
    @test @inferred(permutedims(MMatrix{2,3}(A))) isa MMatrix{3,2}
    @test @inferred(permutedims(MMatrix{2,3}(A))) == A'
    @test @inferred(permutedims(SizedMatrix{2,3}(A))) isa SizedMatrix{3,2}
    @test @inferred(permutedims(SizedMatrix{2,3}(A))) == A'
end

@testset "vcat() and hcat()" begin
    @test @inferred(vcat(SVector(1,2,3))) === SVector(1,2,3)
    @test @inferred(hcat(SVector(1,2,3))) === SMatrix{3,1}(1,2,3)
    @test @inferred(hcat(SMatrix{3,1}(1,2,3))) === SMatrix{3,1}(1,2,3)

    @test @inferred(vcat(SVector(1,2,3), SVector(4,5,6))) === SVector(1,2,3,4,5,6)
    @test @inferred(hcat(SVector(1,2,3), SVector(4,5,6))) === @SMatrix [1 4; 2 5; 3 6]
    @test_throws DimensionMismatch vcat(SVector(1,2,3), @SMatrix [1 4; 2 5])
    @test_throws DimensionMismatch hcat(SVector(1,2,3), SVector(4,5))

    @test @inferred(vcat(@SMatrix([1;2;3]), SVector(4,5,6))) === @SMatrix([1;2;3;4;5;6])
    @test @inferred(vcat(SVector(1,2,3), @SMatrix([4;5;6]))) === @SMatrix([1;2;3;4;5;6])
    @test @inferred(hcat(@SMatrix([1;2;3]), SVector(4,5,6))) === @SMatrix [1 4; 2 5; 3 6]
    @test @inferred(hcat(SVector(1,2,3), @SMatrix([4;5;6]))) === @SMatrix [1 4; 2 5; 3 6]

    @test @inferred(vcat(@SMatrix([1;2;3]), @SMatrix([4;5;6]))) === @SMatrix([1;2;3;4;5;6])
    @test @inferred(hcat(@SMatrix([1;2;3]), @SMatrix([4;5;6]))) === @SMatrix [1 4; 2 5; 3 6]

    @test @inferred(vcat(SVector(1),SVector(2),SVector(3),SVector(4))) === SVector(1,2,3,4)
    @test @inferred(hcat(SVector(1),SVector(2),SVector(3),SVector(4))) === SMatrix{1,4}(1,2,3,4)

    vcat(SVector(1.0f0), SVector(1.0)) === SVector(1.0, 1.0)
    hcat(SVector(1.0f0), SVector(1.0)) === SMatrix{1,2}(1.0, 1.0)

    # issue #388
    let x = SVector(1, 2, 3)
        # current limit: 34 arguments
        hcat(
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
        allocs = @allocated hcat(
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
        @test allocs == 0
        vcat(
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
        allocs = @allocated vcat(
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
        @test allocs == 0
    end

    # issue #561
    let A = Diagonal(SVector(1, 2)), B = @SMatrix [3 4; 5 6]
        @test @inferred(hcat(A, B)) === SMatrix{2, 4}([Matrix(A) Matrix(B)])
    end

    let A = Transpose(@SMatrix [1 2; 3 4]), B = Adjoint(@SMatrix [5 6; 7 8])
        @test @inferred(hcat(A, B)) === SMatrix{2, 4}([Matrix(A) Matrix(B)])
    end

    let A = Diagonal(SVector(1, 2)), B = @SMatrix [3 4; 5 6]
        @test @inferred(vcat(A, B)) === SMatrix{4, 2}([Matrix(A); Matrix(B)])
    end

    let A = Transpose(@SMatrix [1 2; 3 4]), B = Adjoint(@SMatrix [5 6; 7 8])
        @test @inferred(vcat(A, B)) === SMatrix{4, 2}([Matrix(A); Matrix(B)])
    end

    # hcat/vcat + mixtures of Number and SVector / SMatrix
    @test @inferred(vcat(SA[1,2,3], 4, 5, 6)) === SVector{6}((1,2,3,4,5,6))
    @test @inferred(vcat(0, SA[1,2,3])) === SVector{4}((0,1,2,3))
    @test @inferred(hcat(SMatrix{1,3}((1,2,3)), 4, 5, 6)) === SMatrix{1,6}((1,2,3,4,5,6))
    @test @inferred(hcat(0, SMatrix{1,3}((1,2,3)))) === SMatrix{1,4}((0,1,2,3))
    @test @inferred(vcat(MVector((1,2,3)), 4, 5, 6))::MVector == [1,2,3,4,5,6]

    @test @inferred(vcat(SA[1,2,3])) === SA[1,2,3]
    @test @inferred(vcat(SA[1 2 3])) === SA[1 2 3]
    @test @inferred(hcat(SA[1,2,3])) === SMatrix{3,1}(1,2,3)
    @test @inferred(hcat(SA[1 2 3])) === SA[1 2 3]
end

@testset "Base.rest" begin
    x = SA[1, 2, 3]
    @test Base.rest(x) == x
    a, b... = x
    @test b == SA[2, 3]

    x = SA[1 2; 3 4]
    @test Base.rest(x) == vec(x)
    a, b... = x
    @test b == SA[3, 2, 4]

    a, b... = SA[1]
    @test b == []
    @test b isa SVector{0}

    for (Vec, Mat) in [(MVector, MMatrix), (SizedVector, SizedMatrix)]
        x = Vec(1, 2, 3)
        @test Base.rest(x) == x
        @test pointer(Base.rest(x)) != pointer(x)
        a, b... = x
        @test b == Vec(2, 3)

        x = Mat{2,2}(1, 2, 3, 4)
        @test Base.rest(x) == vec(x)
        @test pointer(Base.rest(x)) != pointer(x)
        a, b... = x
        @test b == Vec(2, 3, 4)

        a, b... = Vec(1)
        @test b == []
        @test b isa Vec{0}
    end
end

@testset "zeros/ones/fill" begin
    for ax in ((SOneTo(2),), (SOneTo(2),SOneTo(3)))
        @test @inferred(fill(:abc, ax...)) === @inferred(fill(:abc, ax))
        @test fill(:abc, ax) == fill(:abc, length.(ax)) == fill(:abc, Base.OneTo.(length.(ax)))
        for fz in (zeros, ones)
            @test @inferred(fz(Float32, ax...)) === @inferred(fz(Float32, ax))
            @test fz(Float32, ax) == fz(Float32, length.(ax)) == fz(Float32, Base.OneTo.(length.(ax)))
        end
    end
end
