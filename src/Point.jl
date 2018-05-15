# This file introduces the concept of an "AbstractPoint", which is a
# generalizatoin of an Affine space where you can add vectors to points and
# subtract points to get displacement vectors, but you may not directly add
# points, etc.
#
# Points should have a subset of vector behaviour, including: scalar indexing,
# subtraction two points (giving a vector), addition or subtraction by a vector.

"""
    abstract AbstractPoint{N, T}

An `AbstractPoint` represents the location of a point in an `N`-dimensional
space. In general this space is not assumed to be Cartesian, and the "origin" of
the space (where the coordinates equal zero) is not assumed to have any special
signficance.

However, `AbstractPoint`s do have a well-defined tangent space about their
location, and another point can created by adding a vector to it. Similarly,
two points may be subtracted to result in a displacement vector. Whether or not
displacements are allowed to be large or small is an implementation detail -
however, it is assumed that infinitesimals such as vectors of dual numbers can
be added and subtracted from the point, corresponding to the point's tangent
space.

Points may also carry extra information beyond their continuous cooridinates,
such as discrete data indicating the points location (like a zone number) or
a property of the point. Custom point types derived from `FieldPoint` or
`IndexingPoint` will automatically have this extra information copied - but other
implementations should specialize the method
`similar(p::AbstractPoint{N,T1}, NTuple{N,T2})` to copy over that extra
information.

(see also `Point`, `FieldPoint`, `IndexingPoint`)
"""
abstract AbstractPoint{N, T}

# Base has: `eltype(t::DataType) = eltype(supertype(t))`
Base.eltype{N,T}(::Union{AbstractPoint{N, T}, Type{AbstractPoint{N, T}}}) = T

Base.size(::Type{Any}) = error("size not defined")
Base.size(t::DataType) = size(supertype(t))
@pure Base.size{N,T}(::Union{AbstractPoint{N, T}, Type{AbstractPoint{N, T}}}) = (N,)

Base.length(::Type{Any}) = error("length not defined")
Base.length(t::DataType) = length(supertype(t))
@pure Base.length{N,T}(::Union{AbstractPoint{N, T}, Type{AbstractPoint{N, T}}}) = N

@inline similar{N,T1,T2}(p::AbstractPoint{N,T1}, t::NTuple{N,T2}) = similar_type(p, T2)(t)

@generated function similar_type{P<:AbstractPoint,T}(::Union{P,Type{P}}, ::Type{T})
    # This function has a strange error (on tests) regarding double-inference, if it is marked @pure
    if T == eltype(P)
        return P
    end

    primary_type = (P.name.primary)
    eltype_param_number = super_eltype_param_point(primary_type)
    if isnull(eltype_param_number)
        error("Cannot change $P to take element type $T")
    end

    T_out = primary_type
    for i = 1:length(T_out.parameters)
        if i == get(eltype_param_number)
            T_out = T_out{T}
        else
            T_out = T_out{SA.parameters[i]}
        end
    end
    return T_out
end

@pure function super_eltype_param_point(T)
    T_super = supertype(T)
    if T_super == Any
        error("Unknown error")
    end

    if T_super.name.name == :AbstractPoint
        if isa(T_super.parameters[2], TypeVar)
            tv = T_super.parameters[2]
            for i = 1:length(T.parameters)
                if tv == T.parameters[i]
                    return Nullable{Int}(i)
                end
            end
            error("Unknown error")
        end
        return Nullable{Int}()
    else
        param_number = super_eltype_param(T_super)
        if isnull(param_number)
            return Nullable{Int}()
        else
            tv = T_super.parameters[get(param_number)]
            for i = 1:length(T.parameters)
                if tv == T.parameters[i]
                    return Nullable{Int}(i)
                end
            end
            error("Unknown error")
        end
    end
end

@generated function Tuple{N}(p::AbstractPoint{N})
    exprs = [:(p[$i]) for i = 1+N]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

# Back-and-forth with StaticVectors
@inline (::Type{SV}){SV<:StaticVector}(p::AbstractPoint) = SV(Tuple(p))
@inline (::Type{P}){P<:AbstractPoint}(v::StaticVector) = P(Tuple(v))

+(::AbstractPoint, ::AbstractPoint) = error("Cannot add two points. Consider adding a vector instead.")
-(::AbstractPoint) = error("Cannot take the negative of a point. Consider subtracting it from another point.")
-(::AbstractVector, ::AbstractPoint) = error("Cannot take the negative of a point. Consider subtracting it from another point.")
*(::AbstractPoint, ::Number) = error("Cannot scale a point.")
*(::Number, ::AbstractPoint) = error("Cannot scale a point.")
/(::AbstractPoint, ::Number) = error("Cannot scale a point.")
*(::AbstractMatrix, ::AbstractPoint) = error("Cannot perform matrix multiplication on a point.")

@generated function -(p1::AbstractPoint, p2::AbstractPoint)
    N = length(p1)

    if p1.name.primary != p2.name.primary || length(p2) != N
        error("Cannot subtract a $p2 from a $p1")
    end

    T = promote_type(eltype(p1), eltype(p2))
    newtype = SVector{N, T}

    exprs = [:(p1[$j] - p2[$j]) for j = 1:N]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function -(p::AbstractPoint, v::StaticVector)
    N = length(p)

    if length(v) != N
        error("Dimension mismatch")
    end

    exprs = [:(p[$j] - v[$j]) for j = 1:N]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, similar, :p, Expr(:tuple, exprs...)))
    end
end

@generated function +(p::AbstractPoint, v::StaticVector)
    N = length(p)

    if length(v) != N
        error("Dimension mismatch")
    end

    exprs = [:(p[$j] + v[$j]) for j = 1:N]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, similar, :p, Expr(:tuple, exprs...)))
    end
end

@generated function +(v::StaticVector, p::AbstractPoint)
    N = length(p)

    if length(v) != N
        error("Dimension mismatch")
    end

    T = promote_type(eltype(p1), eltype(p2))
    newtype = SVector{N, T}

    exprs = [:(v[$j] + p[$j]) for j = 1:N]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, similar, :p, Expr(:tuple, exprs...)))
    end
end


"""
    abstract FieldPoint{N,T} <: AbstractPoint{N,T}

A point that indexes its fields, so that `p[i] = getfield(p, i)`.
"""

abstract FieldPoint{N,T} <: AbstractPoint{N,T}

# Is this a good idea?? Should people just define constructors that accept tuples?
@inline (::Type{FP}){FP<:FieldPoint}(x::Tuple) = FP(x...)

@inline getindex(v::FieldPoint, i::Integer) = getfield(v, i)
@inline setindex!(v::FieldPoint, x, i::Integer) = setfield!(v, i, x)

@generated function Tuple{N}(p::FieldPoint{N})
    exprs = [:(getfield(p, $i)) for i = 1+N]

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

@generated function similar{N,T1,T2}(p::FieldPoint{N,T1}, t::NTuple{N,T2})
    if T1 == T2
        newtype = p
    else
        newtype = similar_type(p, T2)
    end

    exprs1 = [:(t[$i]) for i = 1:N]
    exprs2 = [:(getfield(p, $i)) for i = N+1:length(fieldnames(t))]
    exprs = vcat(exprs1, exprs2)

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, exprs...))
    end
end


"""
    abstract IndexingPoint{N,T} <: AbstractPoint{N,T}

A point that delegates indexing to its first field, so that `p[i] = getfield(p,1)[i]`.
"""
abstract IndexingPoint{N,T} <: AbstractPoint{N,T}

@inline Tuple(p::IndexingPoint) = Tuple(getfield(p, 1))

@propagate_inbounds getindex(v::IndexingPoint, i::Integer) = getfield(v, 1)[i]
@propagate_inbounds setindex!(v::IndexingPoint, x, i::Integer) = setindex!(getfield(v, 1), i, x)

@generated function similar{N,T1,T2}(p::IndexingPoint{N,T1}, t::NTuple{N,T2})
    if T1 == T2
        newtype = p
    else
        newtype = similar_type(p, T2)
    end

    exprs1 = [:(t)]
    exprs2 = [:(getfield(p, $i)) for i = 2:length(fieldnames(t))]
    exprs = vcat(exprs1, exprs2)

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, newtype, exprs...))
    end
end


"""
    immutable Point{N, T} <: IndexingPoint{N, T}
    Point(x1, x2, ...)

The coodinates of a point in an N-dimensional Affine space. Compared to a
vector in a Cartesian space, a `Point` has less valid operations since the
origin of the coordinate system is not assumed to have any meaning. The
consequence of this is that there are less operations defined on `Point`s
compared to vectors: you may subtract two points to obtain a dispacement vector,
or add/subtract displacement vectors to your points, but you may not add two
points or scale them.
"""
immutable Point{N, T} <: IndexingPoint{N, T}
    data::NTuple{N, T}
end

@inline (::Type{Point}){N}(t::NTuple{N}) = Point{N}(t)
@inline (::Type{Point{N}}){N, T <: Tuple}(x::T) = Point{N,promote_tuple_eltype(T)}(x)

# Automatically wrap inputs into a Tuple, while avoiding splatting penalties.
@inline (::Type{P}){P<:Point}(x1,x2) = P((x1,x2))
@inline (::Type{P}){P<:Point}(x1,x2,x3) = P((x1,x2,x3))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4) = P((x1,x2,x3,x4))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5) = P((x1,x2,x3,x4,x5))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6) = P((x1,x2,x3,x4,x5,x6))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7) = P((x1,x2,x3,x4,x5,x6,x7))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8) = P((x1,x2,x3,x4,x5,x6,x7,x8))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15))
@inline (::Type{P}){P<:Point}(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16) = P((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16))

export AbstractPoint, FieldPoint, IndexingPoint, Point
