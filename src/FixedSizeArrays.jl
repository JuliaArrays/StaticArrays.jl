"""
Compatibility layer for transferring from FixedSizeArrays. This provides
alternative definitions of `Vec`, `Mat`, `Point`, `FixedVectorNoTuple`, `@fsa`,
etc, using StaticArrays as a backend.

The type definitions are not "perfect" matches because the type parameters are
different. However, it should cover common method signatures and constructors.
"""
module FixedSizeArrays

using ..StaticArrays

export FixedArray
export FixedVector
export FixedMatrix
export Mat, Vec, Point
export @fsa
export FixedVectorNoTuple

const FixedArray = StaticArray
const FixedVector = StaticVector
const FixedMatrix = StaticMatrix
const Vec = SVector
const Mat = SMatrix
const FixedVectorNoTuple = FieldVector

macro fsa(ex)
    @assert isa(ex, Expr)
    if ex.head == :vect # Vector
        return Expr(:call, SVector{length(ex.args)}, Expr(:tuple, ex.args...))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return Expr(:call, SMatrix{s1, s2}, Expr(:tuple, ex.args...))
    elseif ex.head == :vcat
        if isa(ex.args[1], Expr) && ex.args[1].head == :row # n x m
            # Validate
            s1 = length(ex.args)
            s2s = map(i -> ((isa(ex.args[i], Expr) && ex.args[i].head == :row) ? length(ex.args[i].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i].args[j] for i = 1:s1, j = 1:s2]
            return Expr(:call, SMatrix{s1, s2}, Expr(:tuple, exprs...))
        else # n x 1
            return Expr(:call, SMatrix{length(ex.args), 1}, Expr(:tuple, ex.args...))
        end
    end
end

###########
## Point ##
###########

immutable Point{S, T} <: StaticVector{T}
    data::NTuple{S, T}
end

@inline (::Type{Point}){S}(x::NTuple{S}) = Point{S}(x)
@inline (::Type{Point{S}}){S, T}(x::NTuple{S,T}) = Point{S,T}(x)
@inline (::Type{Point{S}}){S, T <: Tuple}(x::T) = Point{S,promote_tuple_eltype(T)}(x)

Base.@pure Base.size{S}(::Union{Point{S},Type{Point{S}}}) = (S, )
Base.@pure Base.size{S,T}(::Type{Point{S,T}}) = (S,)

Base.@propagate_inbounds function Base.getindex(v::Point, i::Integer)
    v.data[i]
end

@inline Base.Tuple(v::Point) = v.data

end
