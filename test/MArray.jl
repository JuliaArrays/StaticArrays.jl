@testset "MArray" begin
    @testset "Inner Constructors" begin
        @test MArray{(1,),Int,1,1}((1,)).data === (1,)
        @test MArray{(1,),Float64,1,1}((1,)).data === (1.0,)
        @test MArray{(2,2),Float64,2,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(MArray{(1,),Int,1,1}(), MArray{(1,),Int,1,1})
        @test isa(MArray{(1,),Int,1}(), MArray{(1,),Int,1,1})
        @test isa(MArray{(1,),Int}(), MArray{(1,),Int,1,1})

        # Bad input
        @test_throws Exception MArray{(2,),Int,1,2}((1,))
        @test_throws Exception MArray{(1,),Int,1,1}(())

        # Bad parameters
        @test_throws Exception MArray{(1,),Int,1,2}((1,))
        @test_throws Exception MArray{(1,),Int,2,1}((1,))
        @test_throws Exception MArray{(1,),1,1,1}((1,))
        @test_throws Exception MArray{(2,),Int,1,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test MArray{(1,),Int,1}((1,)).data === (1,)
        @test MArray{(1,),Int}((1,)).data === (1,)
        @test MArray{(1,)}((1,)).data === (1,)

        @test MArray{(2,2),Int,2}((1,2,3,4)).data === (1,2,3,4)
        @test MArray{(2,2),Int}((1,2,3,4)).data === (1,2,3,4)
        @test MArray{(2,2)}((1,2,3,4)).data === (1,2,3,4)

        @test ((@MArray [1])::MArray{(1,)}).data === (1,)
        @test ((@MArray [1,2])::MArray{(2,)}).data === (1,2)
        @test ((@MArray Float64[1,2,3])::MArray{(3,)}).data === (1.0, 2.0, 3.0)
        @test ((@MArray [1 2])::MArray{(1,2)}).data === (1, 2)
        @test ((@MArray Float64[1 2])::MArray{(1,2)}).data === (1.0, 2.0)
        @test ((@MArray [1 ; 2])::MArray{(2,1)}).data === (1, 2)
        @test ((@MArray Float64[1 ; 2])::MArray{(2,1)}).data === (1.0, 2.0)
        @test ((@MArray [1 2 ; 3 4])::MArray{(2,2)}).data === (1, 3, 2, 4)
        @test ((@MArray Float64[1 2 ; 3 4])::MArray{(2,2)}).data === (1.0, 3.0, 2.0, 4.0)

        @test (ex = macroexpand(:(@MArray [1 2; 3])); isa(ex, Expr) && ex.head == :error)
    end

    @testset "Methods" begin
        m = @MArray [11 13; 12 14]

        @test isimmutable(m) == false

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @test Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(MArray{(2,2),Int,2}) === (2, 2)
        @test size(MArray{(2,2),Int}) === (2, 2)
        @test size(MArray{(2,2)}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4
    end

    @testset "setindex!" begin
        v = @MArray [1,2,3]
        v[1] = 11
        v[2] = 12
        v[3] = 13
        @test v.data === (11, 12, 13)

        m = @MArray [0 0; 0 0]
        m[1] = 11
        m[2] = 12
        m[3] = 13
        m[4] = 14
        @test m.data === (11, 12, 13, 14)
    end
end
