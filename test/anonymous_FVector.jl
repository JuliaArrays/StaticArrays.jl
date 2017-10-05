using StaticArrays
using Base.Test

a = @FVector Float64 [a,b,c]
b = @FVector Float64 [a,b,c] [1,2,3]

c = @MFVector Float64 [a,b,c]
d = @MFVector Float64 [a,b,c] [1,2,3]

@test_throws ErrorException fill!(a,1)
@test typeof(b) <: FieldVector{3,Float64}
b.+b
c.=2d
@test c == [2,4,6]
@test typeof(c) <: FieldVector{3,Float64}
c.+d
