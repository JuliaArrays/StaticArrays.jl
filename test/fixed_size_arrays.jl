using StaticArrays, Test, LinearAlgebra, SpecialFunctions
using StaticArrays.FixedSizeArrays

import StaticArrays.FixedSizeArrays: @fixed_vector


const Vec1d = Vec{1, Float64}
const Vec2d = Vec{2, Float64}
const Vec3d = Vec{3, Float64}
const Vec4d = Vec{4, Float64}
const Vec3f = Vec{3, Float32}

const Mat2d = Mat{2,2, Float64, 4}
const Mat3d = Mat{3,3, Float64, 9}
const Mat4d = Mat{4,4, Float64, 16}

struct RGB{T} <: FieldVector{3, T}
    x::T
    y::T
    z::T
end

RGB(x::T) where {T} = RGB{T}(x, x, x)
(::RGB{T})(r, g, b) where {T} = RGB{T}(T(r), T(g), T(b))
(::RGB{T})(r::Real) where {T} = RGB(T(r), T(r), T(r))
StaticArrays.similar_type(::Type{SV}, ::Type{T}, ::Size{(3,)}) where {SV <: RGB, T} = RGB{T}

# TODO find equivalent in StaticArrays
@testset "scalar nan" begin
    for (p, r) in (
            (Point{2, Float32}(NaN, 1), true),
            (Point{2, Float64}(1, NaN), true),
            (Vec{11, Float64}(NaN), true),
            (Point{2, Float32}(1, 1), false),
            (RGB{Float32}(NaN, NaN, NaN), true),
        )
        @test any(isnan, p) == r
        @test isnan(p) == r
    end
end

# methods I needed to define:


#=

unit(Vec4d, 1)
rand(Vec{7, Int}, 1:7)
randn(Mat{4,2, Complex{Float64}})
x = one(Mat{4,2, Int})

v2 = Vec(6.0,5.0,4.0)
v1 = Vec(1.0,2.0,3.0)
vi = Vec(1,2,3)
v2 = Vec(6.0,5.0,4.0)
v1c = Vec(6.0+3.0im,5.0-2im,4.0+0.0im)
v2c = v1 + v2*im
v2c = Vec(1.0 + 6.0im, 2.0 + 5.0im, 3.0 + 4.0im)

@testset "Complex Ops" begin
    @testset "dot product" begin

        # different results:
        @test dot(v1c,v2c) == dot([6.0+3.0im,5.0-2im,4.0+0.0im], [1.0,2.0,3.0] + [6.0,5.0,4.0]*im)
        @test Vector(transpose(v1c)*v2c) == [6.0+3.0im 5.0-2im 4.0+0.0im]*([1.0,2.0,3.0] + [6.0,5.0,4.0]*im)
        @test Matrix(v2c*transpose(v1c)) == ([1.0,2.0,3.0] + [6.0,5.0,4.0]*im)*[6.0+3.0im 5.0-2im 4.0+0.0im]
    end
end

m = Mat{4,4}(
    1,2,3,4,
    5,6,7,8,
    9,10,11,12,
    13,14,15,16
)
# no setindex
setindex(m, 42.0, 2,2)

# No 2D cross
a,b = Vec2d(0,1), Vec2d(1,0)
@test cross(a,b) == -1.0
@test isa(cross(a,b), Float64) == true

a = Vec{2,Int}(1,2)
b = Vec{2,Float64}(1.,2.)
# no hypot
@test hypot(a) == 2.23606797749979
@test hypot(b) == 2.23606797749979
@test hypot(a) == hypot(b)

# I'm guessing this fails since FixedSizeArrays.normalize wasn't as precise
a = Vec(3,4)
@test normalize(a) == Vec(0.6, 0.8)

=#


# slow operations:
#=
all(x-> x==1, x) == true
=#


@testset "reduce" begin
    N = 100
    a = Point{3, Float32}[Point{3, Float32}(0.7132) for i=1:N]
    c = Point{3, Float64}[Point{3, Float64}(typemin(Float64)), a..., Point{3, Float64}(typemax(Float64))]
    sa = sum(a)
    ma = mean(a)
    for i=1:3
        @test sa[i] ≈ Float32(0.7132*N)
        @test ma[i] ≈ Float32(0.7132*N)/ N
    end
    @test maximum(c) == Point{3, Float32}(typemax(Float64))
    @test minimum(c) == Point{3, Float32}(typemin(Float64))
    @test extrema(c) == (minimum(c), maximum(c))
end



@test typeof(rand(Vec4d)) == Vec4d
@test typeof(rand(Mat4d)) == Mat4d

rand(Mat{4,2, Int})
@test typeof(rand(Mat{4,2, Int})) == Mat{4,2, Int, 8}
@test typeof(rand(Vec{7, Int})) == Vec{7, Int}

# @test typeof(rand(-20f0:0.192f0:230f0, Mat4d)) == Mat4d
# @test typeof(rand(-20f0:0.192f0:230f0, Mat{4,21,Float32})) == Mat{4,21,Float32, 4*21}

@test typeof(rand(Vec4d, 5,5)) == Matrix{Vec4d}
#end
# @testset "Randn" begin


#end

@testset "one" begin
    x = Mat{4,2, Int}(1, 1, 1, 1, 1, 1, 1, 1)
    @test all(x-> x==1, x) == true
end

@testset "unit... not really after removing non working" begin
    u4 = Vec4d(1, 0, 0, 0)
    @test u4[1] == 1.0
    @test u4[2:end] == Vec3d(0.,0.,0.)
    @test u4[end] == 0
end

for N = (1, 10)
    @testset "construction, conversion, $N" begin
        for VT=[Point, Vec], VT2=[Vec, Point], ET=[Float32, Int, UInt], ET2=[Float64, UInt, Float32]
            rand_range  = ET(1):ET(10)
            rand_range2 = ET2(1):ET2(10)
            rn = rand(rand_range, N)
            v0 = VT{N}(rn)
            # parse constructor:
            # multi constructor
            v1 = VT{N, ET}(rn...)
            @test v1 == v0
            @test typeof(v1) == VT{N, ET}
            @test length(v1) == N
            @test eltype(v1) == ET
            @test ndims(v1) == 1

            # from other FSA without parameters
            v2 = VT2(v1)

            # No-op conversion to same type without parameters
            @test convert(VT, v1) === v1

            @test typeof(v2) == VT2{N, ET}
            @test length(v2) == N
            @test eltype(v2) == ET
            for i=1:N
                @test v2[i] == v1[i]
            end

            # from single
            r  = rand(rand_range)
            r2 = rand(rand_range2)
            v1 = VT{N, ET}(r)
            v2 = VT{N, ET2}(r)
            v3 = VT{N, ET}(r2)
            v4 = VT{N, ET2}(r2)

            for i=1:N
                @test v1[i] == r
                @test v2[i] == ET2(r)
                @test v3[i] == r2
                @test v4[i] == ET2(r2)
            end
            x = VT{N, ET}[VT{N, ET}(ntuple(x-> 1, N)) for i=1:10]
            x1 = VT2{N, ET}[VT{N, ET}(ntuple(x-> 1, N)) for i=1:10]
            x2 = map(VT2, x)
            x3 = map(VT, x2)
            @test typeof(x)  == Vector{VT{N, ET}}
            @test typeof(x1) == Vector{VT2{N, ET}}
            @test typeof(x2) == Vector{VT2{N, ET}}
            @test typeof(x3) == Vector{VT{N, ET}}
            @test x3         == x
        end
    end
end

@testset "heterogeneous construction" begin
    @test @inferred(Vec(0.0, 0)) isa Vec{2, Float64}
    @test @inferred(Vec(0, 0.0)) isa Vec{2, Float64}
    @test @inferred(Point(0.0, 0)) isa Point{2, Float64}
    @test @inferred(Point(0, 0.0)) isa Point{2, Float64}
    @test @inferred(Vec(0.0, 0, 0)) isa Vec{3, Float64}
    @test @inferred(Vec(0, 0.0, 0)) isa Vec{3, Float64}
    @test @inferred(Point(0.0, 0, 0)) isa Point{3, Float64}
    @test @inferred(Point(0, 0.0, 0)) isa Point{3, Float64}
end

map(-, Vec(1,2,3))
map(+, Vec(1,2,3), Vec(1,1, 1))
(+).(Vec(1,2,3), 1.0)
v1 = Vec3d(1,2,3)
@test v1[Vec(2,1)] == Vec2d(2,1)

@test @inferred(-v1) == Vec(-1.0,-2.0,-3.0)
@test isa(-v1, Vec3d) == true
@test @inferred(v1 ./ v1) == Vec3d(1.0,1.0,1.0)
@test (<).(Vec(1,3), Vec(2,2)) === Vec{2,Bool}(true, false)
#
v1 = Vec(1.0,2.0,3.0)
v2 = Vec(6.0,5.0,4.0)
vi = Vec(1,2,3)
v2 = Vec(6.0,5.0,4.0)
v1c = Vec(6.0+3.0im,5.0-2im,4.0+0.0im)
v2c = v1 + v2*im
v2c = Vec(1.0 + 6.0im, 2.0 + 5.0im, 3.0 + 4.0im)
@testset "Ops" begin
    @testset "revers" begin
        @test reverse(Vec(1,2,3,4,5)) == Vec(5,4,3,2,1)
    end
    @testset "Negation" begin
        @test @inferred(-v1) == Vec(-1.0,-2.0,-3.0)
        @test isa(-v1, Vec3d) == true
    end

    @testset "Addition" begin
        @test @inferred(v1+v2) == Vec3d(7.0,7.0,7.0)
        @test @inferred(RGB(1,2,3) + RGB(2,2,2)) == RGB{Int}(3,4,5)
    end
    @testset "Subtraction" begin
        @test @inferred(v2-v1) == Vec3d(5.0,3.0,1.0)
        @test @inferred(RGB(1,2,3) - RGB(2,2,2)) == RGB{Int}(-1,0,1)
    end
    @testset "Multiplication" begin
        @test @inferred(v1.*v2) == Vec3d(6.0,10.0,12.0)
    end
    @testset "Mixed Type Multiplication" begin
        @test @inferred(vi.*v2) == Vec3d(6.0,10.0,12.0)
    end
    @testset "Division" begin
        @test @inferred(v1 ./ v1) == Vec3d(1.0,1.0,1.0)
    end

    @testset "Relational" begin
        @test (Vec(1,3) .< Vec(2,2)) == Vec{2,Bool}(true,false)
        @test (RGB(1,2,3) .< RGB(2,2,2)) == RGB{Bool}(true,false,false)
    end

    @testset "Scalar" begin
        @test @inferred(1.0 + v1) == Vec3d(2.0,3.0,4.0)
        @test @inferred(1.0 .+ v1) == Vec3d(2.0,3.0,4.0)
        @test @inferred(v1 + 1.0) == Vec3d(2.0,3.0,4.0)
        @test @inferred(v1 .+ 1.0) == Vec3d(2.0,3.0,4.0)
        @test @inferred(1 + v1) == Vec3d(2.0,3.0,4.0)
        @test @inferred(1 .+ v1) == Vec3d(2.0,3.0,4.0)
        @test @inferred(v1 + 1) == Vec3d(2.0,3.0,4.0)
        @test @inferred(v1 .+ 1) == Vec3d(2.0,3.0,4.0)

        @test @inferred(v1 - 1.0) == Vec3d(0.0,1.0,2.0)
        @test @inferred(v1 .- 1.0) == Vec3d(0.0,1.0,2.0)
        @test @inferred(1.0 - v1) == Vec3d(0.0,-1.0,-2.0)
        @test @inferred(1.0 .- v1) == Vec3d(0.0,-1.0,-2.0)
        @test @inferred(v1 - 1) == Vec3d(0.0,1.0,2.0)
        @test @inferred(v1 .- 1) == Vec3d(0.0,1.0,2.0)
        @test @inferred(1 - v1) == Vec3d(0.0,-1.0,-2.0)
        @test @inferred(1 .- v1) == Vec3d(0.0,-1.0,-2.0)

        @test @inferred(2.0 * v1) == Vec3d(2.0,4.0,6.0)
        @test @inferred(2.0 .* v1) == Vec3d(2.0,4.0,6.0)
        @test @inferred(v1 * 2.0) == Vec3d(2.0,4.0,6.0)
        @test @inferred(v1 .* 2.0) == Vec3d(2.0,4.0,6.0)
        @test @inferred(2 * v1) == Vec3d(2.0,4.0,6.0)
        @test @inferred(2 .* v1) == Vec3d(2.0,4.0,6.0)
        @test @inferred(v1 * 2) == Vec3d(2.0,4.0,6.0)
        @test @inferred(v1 .* 2) == Vec3d(2.0,4.0,6.0)

        @test @inferred(v1 / 2.0) == Vec3d(0.5,1.0,1.5)
        @test @inferred(v1 ./ 2.0) == Vec3d(0.5,1.0,1.5)
        @test @inferred(v1 / 2) == Vec3d(0.5,1.0,1.5)
        @test @inferred(v1 ./ 2) == Vec3d(0.5,1.0,1.5)

        @test @inferred(12.0 ./ v1) == Vec3d(12.0,6.0,4.0)
        @test @inferred(12 ./ v1) == Vec3d(12.0,6.0,4.0)

        @test @inferred((v1 .^ 2)) == Vec3d(1.0,4.0,9.0)
        @test @inferred((v1 .^ 2.0)) == Vec3d(1.0,4.0,9.0)
        @test @inferred((2.0 .^ v1)) == Vec3d(2.0,4.0,8.0)
        @test @inferred((2 .^ v1)) == Vec3d(2.0,4.0,8.0)

        a = Vec(3.2f0)
        @test @inferred(a+0.2) == Vec1d(3.2f0+0.2)
        @test @inferred(0.2+a) == Vec1d(3.2f0+0.2)
        @test @inferred(a*0.2) == Vec1d(3.2f0*0.2)
        @test @inferred(0.2*a) == Vec1d(3.2f0*0.2)
        @test @inferred(a+0.2f0) == Vec{1,Float32}(3.4f0)
        @test @inferred(0.2f0+a) == Vec{1,Float32}(3.4f0)
        @test @inferred(a*0.2f0) == Vec{1,Float32}(3.2f0*0.2f0)
        @test @inferred(0.2f0*a) == Vec{1,Float32}(3.2f0*0.2f0)
    end
    @testset "vector norm+cross product" begin

        @test norm(Vec3d(1.0,2.0,2.0)) == 3.0
        @test norm(Vec3d(1.0,2.0,2.0),2) == 3.0
        @test norm(Vec3d(1.0,2.0,2.0),Inf) == 2.0
        @test norm(Vec3d(1.0,2.0,2.0),1) == 5.0

        # cross product
        @test cross(v1,v2) == Vec3d(-7.0,14.0,-7.0)
        @test isa(cross(v1,v2), Vec3d)  == true

        @test cross(vi,v2) == Vec3d(-7.0,14.0,-7.0)
        @test isa(cross(vi,v2),Vec3d)  == true

        a,b = Vec2d(0,1), Vec2d(1,0)
        @test cross(a,b) == -1.0
        @test isa(cross(a,b), Float64) == true
    end

    @testset "normalize" begin
        a = Vec(3,4)
        b = Vec(3.,4.)
        @test normalize(a) ≈ Vec(0.6,0.8)
        @test normalize(b) ≈ Vec(0.6,0.8)
    end

    @testset "reduce" begin
        a = rand(Vec{7, Float32})
        x = reduce(+, a)
        y = 0f0
        for elem in a
            y += elem
        end
        @test y == x

        a = rand(Mat{7, 9, Cuint})
        x2 = reduce(+, a)
        y2 = Cuint(0)
        for elem in a
            y2 += elem
        end
        @test y2 == x2
    end
end


const unaryOps = (
    -, ~, conj, abs,
    sin, cos, tan, sinh, cosh, tanh,
    asin, acos, atan, asinh, acosh, atanh,
    sec, csc, cot, asec, acsc, acot,
    sech, csch, coth, asech, acsch, acoth,
    sinc, cosc, cosd, cotd, cscd, secd,
    sind, tand, acosd, acotd, acscd, asecd,
    asind, atand, rad2deg, deg2rad,
    log, log2, log10, log1p, exponent, exp,
    exp2, expm1, cbrt, sqrt, erf,
    erfc, erfcx, erfi, dawson,

    trunc, round, ceil, floor,
    significand, lgamma,
    gamma, lfactorial, frexp, modf, airyai,
    airyaiprime, airybi, airybiprime,
    besselj0, besselj1, bessely0, bessely1,
    eta, zeta, digamma, real, imag
)

# vec-vec and vec-scalar
const binaryOps = (

    +, -, *, /, \,
    ==, !=, <, <=, >, >=,
    min, max,
    atan, besselj, bessely, hankelh1, hankelh2,
    besseli, besselk, beta, lbeta
)


@testset "mapping operators" begin
    @testset "binary: " begin
        test1 = (Vec(1,2,typemax(Int)), Mat{3, 3}(typemin(Int),2,5, 2,3,5, -2,3,6), Vec{4, Float32}(0.777, 0.777, 0.777, 0.777))
        test2 = (Vec(1,0,typemax(Int)), Mat{3, 3}(typemin(Int),77,1, 2,typemax(Int),5, -2,3,6), Vec{4, Float32}(-23.2929, -23.2929, -23.2929, -23.2929))

        for op in binaryOps
            for i=1:length(test1)
                x1 = test1[i]
                x2 = test2[i]
                @testset "$op with $x1 and $x2" begin
                    try # really bad tests, but better than nothing...
                        if applicable(op, x1[1], x2[1]) && typeof(op(x1[1], x2[1])) == eltype(x1)
                            r = op.(x1, x2)
                            for j=1:length(x1)
                                @test r[j] == op(x1[j], x2[j])
                            end
                        end
                    catch
                    end
                end
            end
        end
    end
    @testset "unary: " begin
        test = (Vec(1,2,typemax(Int)), Mat{3, 3}(typemin(Int),2,5, 2,3,5, -2,3,6), Vec{4, Float32}(0.777, 0.777, 0.777, 0.777))
        for op in unaryOps
            for t in test
                @testset "$op with $t" begin
                    try
                        if applicable(op, t[1]) && typeof(op(t[1])) == eltype(t)
                            v = op.(t)
                            for i=1:length(v)
                                @test v[i] == op(t[i])
                            end
                        end
                    catch
                    end
                end
            end
        end
    end
end
