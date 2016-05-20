@testset "AbstractArray interface" begin
    @testset "size and length" begin
        vec = [1,2,3,4]
        sv = SVector{(4,)}(vec)
        mv = MVector{(4,)}(vec)

        mat = eye(3)
        sm = MMatrix{(3,3)}(mat)
        mm = MMatrix{(3,3)}(mat)

        @test size(sv) == size(vec)
        @test size(mv) == size(vec)
        @test size(sm) == size(mat)
        @test size(mm) == size(mat)

        @test size(sv,1) == size(vec,1)
        @test size(mv,1) == size(vec,1)
        @test size(sm,1) == size(mat,1)
        @test size(mm,1) == size(mat,1)
        @test size(sv,2) == size(vec,2) # Test for
        @test size(mv,2) == size(vec,2) # trailing 1's
        @test size(sm,2) == size(mat,2)
        @test size(mm,2) == size(mat,2)

        @test length(sv) == length(vec)
        @test length(mv) == length(vec)
        @test length(sm) == length(mat)
        @test length(mm) == length(mat)

        @test size(typeof(sv)) == size(vec)
        @test size(typeof(mv)) == size(vec)
        @test size(typeof(sm)) == size(mat)
        @test size(typeof(mm)) == size(mat)

        @test size(typeof(sv),1) == size(vec,1)
        @test size(typeof(mv),1) == size(vec,1)
        @test size(typeof(sm),1) == size(mat,1)
        @test size(typeof(mm),1) == size(mat,1)
        @test size(typeof(sv),2) == size(vec,2) # Test for
        @test size(typeof(mv),2) == size(vec,2) # trailing 1's
        @test size(typeof(sm),2) == size(mat,2)
        @test size(typeof(mm),2) == size(mat,2)

        @test length(typeof(sv)) == length(vec)
        @test length(typeof(mv)) == length(vec)
        @test length(typeof(sm)) == length(mat)
        @test length(typeof(mm)) == length(mat)
    end

    @testset "Miscellanious introspection" begin
        vec = [1,2,3,4]
        sv = SVector{(4,)}(vec)
        mv = MVector{(4,)}(vec)

        @test Base.isassigned(sv, 4)    == Base.isassigned(vec, 4)
        @test Base.isassigned(sv, 5)    == Base.isassigned(vec, 5)
        @test Base.isassigned(sv, 2, 1) == Base.isassigned(vec, 2, 1) # test for trailing ones
        @test Base.isassigned(mv, 4)    == Base.isassigned(vec, 4)
        @test Base.isassigned(mv, 5)    == Base.isassigned(vec, 5)
        @test Base.isassigned(mv, 2, 1) == Base.isassigned(vec, 2, 1)

        @test Base.linearindexing(sv) == Base.LinearFast()
        @test Base.linearindexing(mv) == Base.LinearFast()
        @test Base.linearindexing(typeof(sv)) == Base.LinearFast()
        @test Base.linearindexing(typeof(mv)) == Base.LinearFast()
    end

    @testset "similar and similar_type" begin
        vec = [1,2,3,4]
        sv = SVector{(4,)}(vec)
        mv = MVector{(4,)}(vec)

        @test_throws Exception similar(sv)
        @test typeof(similar(mv)) == typeof(mv)
        @test_inferred similar(mv)
        @test_throws Exception similar(sv, Float64)
        @test typeof(similar(mv, Float64)) <: MVector{(4,),Float64}
        @test_inferred similar(mv, Float64)
        @test_throws Exception similar(sv, (2,2))
        @test_throws Exception similar(mv, (2,2))
        @test_throws Exception similar(sv, Float64, (2,2))
        @test_throws Exception similar(mv, Float64, (2,2))
        @test_throws Exception similar(sv, Val{(2,2)})
        @test typeof(similar(mv, Val{(2,2)})) <: MArray{(2,2),Int}
        @test_inferred similar(mv, Val{(2,2)})
        @test_throws Exception similar(sv, Float64, Val{(2,2)})
        @test typeof(similar(mv, Float64, Val{(2,2)})) <: MArray{(2,2),Float64}
        @test_inferred similar(mv, Float64, Val{(2,2)})

        @test similar_type(sv) == typeof(sv)
        @test_inferred similar_type(sv)
        @test similar_type(mv) == typeof(mv)
        @test_inferred similar_type(mv)
        @test similar_type(sv, Float64) == SArray{(4,),Float64,1,NTuple{4,Float64}}
        @test_inferred similar_type(sv, Float64)
        @test similar_type(mv, Float64) == MArray{(4,),Float64,1,NTuple{4,Float64}}
        @test_inferred similar_type(mv, Float64)
        @test_throws Exception similar_type(sv, (2,2))
        @test_throws Exception similar_type(mv, (2,2))
        @test_throws Exception similar_type(sv, Float64, (2,2))
        @test_throws Exception similar_type(mv, Float64, (2,2))
        @test similar_type(sv, Val{(2,2)}) == SArray{(2,2),Int,2,NTuple{4,Int}}
        @test_inferred similar_type(sv, Val{(2,2)})
        @test similar_type(mv, Val{(2,2)}) == MArray{(2,2),Int,2,NTuple{4,Int}}
        @test_inferred similar_type(mv, Val{(2,2)})
        @test similar_type(sv, Float64, Val{(2,2)}) == SArray{(2,2),Float64,2,NTuple{4,Float64}}
        @test_inferred similar_type(sv, Float64, Val{(2,2)})
        @test similar_type(mv, Float64, Val{(2,2)}) == MArray{(2,2),Float64,2,NTuple{4,Float64}}
        @test_inferred similar_type(mv, Float64, Val{(2,2)})
    end

    @testset "reshape and permutedims" begin
        vec = [4,5,6,7]
        sv = SVector{(4,)}(vec)
        mv = MVector{(4,)}(vec)

        mat = reshape(vec,(2,2))
        sm = SMatrix{(2,2)}(mat)
        mm = MMatrix{(2,2)}(mat)

        @test reshape(sv, Val{(2,2)}) === sm
        @test_inferred reshape(sv, Val{(2,2)})
        @test reshape(mv, Val{(2,2)}) == mm
        @test_inferred reshape(mv, Val{(2,2)})
        @test isa(reshape(mv, Val{(2,2)}), MMatrix)
        @test_throws Exception reshape(sv, (2,2)) == sm
        @test_throws Exception reshape(sv, (2,2)) == sm

    end
end
