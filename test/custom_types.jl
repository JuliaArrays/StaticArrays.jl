@testset "Custom types" begin
    # Issue 123
    @eval (struct MyType{N, T} <: StaticVector{N, T}
        data::NTuple{N, T}
    end)
    @test (MyType(3, 4) isa MyType{2, Int})

    # Issue 110
    @eval (struct Polly{N,T}
        data::SVector{N,T}
    end)
    @test (Polly{2,Float64}((1.0, 0.0)) isa Polly)
end
