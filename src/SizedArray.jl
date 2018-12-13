
"""
    SizedArray{Tuple{dims...}}(array)

Wraps an `Array` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.

(Also, `Size(dims...)(array)` acheives the same thing)
"""
struct SizedArray{S <: Tuple, T, N, M} <: StaticArray{S, T, N}
    data::Array{T, M}

    function SizedArray{S, T, N, M}(a::Array) where {S, T, N, M}
        if length(a) != tuple_prod(S)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new{S,T,N,M}(a)
    end

    @static if VERSION < v"1.0"
        function SizedArray{S, T, N, M}() where {S, T, N, M}
            Base.depwarn("`SizedArray{S,T,N,M}()` is deprecated, use `SizedArray{S,T,N,M}(undef)` instead", :SizedArray)
            new{S, T, N, M}(Array{T, M}(undef, S.parameters...))
        end
    end

    function SizedArray{S, T, N, M}(::UndefInitializer) where {S, T, N, M}
        new{S, T, N, M}(Array{T, M}(undef, S.parameters...))
    end
end

@inline SizedArray{S,T,N}(a::Array{T,M}) where {S,T,N,M} = SizedArray{S,T,N,M}(a)
@inline SizedArray{S,T}(a::Array{T,M}) where {S,T,M} = SizedArray{S,T,tuple_length(S),M}(a)
@inline SizedArray{S}(a::Array{T,M}) where {S,T,M} = SizedArray{S,T,tuple_length(S),M}(a)

@inline SizedArray{S,T,N}(::UndefInitializer) where {S,T,N} = SizedArray{S,T,N,N}(undef)
@inline SizedArray{S,T}(::UndefInitializer) where {S,T} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(undef)

@static if VERSION < v"1.0"
    @inline function SizedArray{S,T,N}() where {S,T,N}
        Base.depwarn("`SizedArray{S,T,N}()` is deprecated, use `SizedArray{S,T,N}(undef)` instead", :SizedArray)
        SizedArray{S,T,N,N}(undef)
    end
    @inline function SizedArray{S,T}() where {S,T}
        Base.depwarn("`SizedArray{S,T}()` is deprecated, use `SizedArray{S,T}(undef)` instead", :SizedArray)
        SizedArray{S,T,tuple_length(S),tuple_length(S)}(undef)
    end
end

@generated function (::Type{SizedArray{S,T,N,M}})(x::NTuple{L,Any}) where {S,T,N,M,L}
    if L != tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = SizedArray{S,T,N,M}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline SizedArray{S,T,N}(x::Tuple) where {S,T,N} = SizedArray{S,T,N,N}(x)
@inline SizedArray{S,T}(x::Tuple) where {S,T} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(x)
@inline SizedArray{S}(x::NTuple{L,T}) where {S,T,L} = SizedArray{S,T,tuple_length(S),tuple_length(S)}(x)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::SizedArray) where {SA<:SizedArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
@inline Array(sa::SizedArray) = sa.data
@inline Array{T}(sa::SizedArray{S,T}) where {T,S} = sa.data
@inline Array{T,N}(sa::SizedArray{S,T,N}) where {T,S,N} = sa.data

@inline convert(::Type{Array}, sa::SizedArray) = sa.data
@inline convert(::Type{Array{T}}, sa::SizedArray{S,T}) where {T,S} = sa.data
@inline convert(::Type{Array{T,N}}, sa::SizedArray{S,T,N}) where {T,S,N} = sa.data

@propagate_inbounds getindex(a::SizedArray, i::Int) = getindex(a.data, i)
@propagate_inbounds setindex!(a::SizedArray, v, i::Int) = setindex!(a.data, v, i)

SizedVector{S,T,M} = SizedArray{Tuple{S},T,1,M}
@inline SizedVector{S}(a::Array{T,M}) where {S,T,M} = SizedArray{Tuple{S},T,1,M}(a)
@inline SizedVector{S}(x::NTuple{L,T}) where {S,T,L} = SizedArray{Tuple{S},T,1,1}(x)

SizedMatrix{S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M}
@inline SizedMatrix{S1,S2}(a::Array{T,M}) where {S1,S2,T,M} = SizedArray{Tuple{S1,S2},T,2,M}(a)
@inline SizedMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = SizedArray{Tuple{S1,S2},T,2,2}(x)

if isdefined(Base, :dataids) # v0.7-
    Base.dataids(sa::SizedArray) = Base.dataids(sa.data)
end

"""
    Size(dims)(array)

Creates a `SizedArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S})(a::Array) where {S} = SizedArray{Tuple{S...}}(a)


function promote_rule(::Type{<:SizedArray{S,T,N,M}}, ::Type{<:SizedArray{S,U,N,M}}) where {S,T,U,N,M}
    SizedArray{S,promote_type(T,U),N,M}
end
