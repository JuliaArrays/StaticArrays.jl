@testset "Custom types" begin
    # Issue 123
    @eval (struct MyType{N, T} <: StaticVector{T}
        data::NTuple{N, T}
    end)
    @test (MyType(3, 4) isa MyType{2, Int})
end
