# Test for https://github.com/JuliaArrays/StaticArrays.jl/issues/228
module PromotionTest
    struct Foo
        x::Int
    end

    struct Bar{S}
        x::Int
    end

    Base.promote_rule(::Type{Bar{S1}}, ::Type{Bar{S2}}) where {S1, S2} = Foo
end

@testset "test for side-effects of promote_tuple_eltype" begin
    b1 = PromotionTest.Bar{:x}(1)
    b2 = PromotionTest.Bar{:y}(2)
    @test @inferred(StaticArrays.promote_tuple_eltype((b1, b2))) == PromotionTest.Foo
    @test promote_type(typeof(b1), typeof(b2)) == PromotionTest.Foo
end
