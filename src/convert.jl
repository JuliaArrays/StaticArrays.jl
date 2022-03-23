# A help wrapper to distinguish `SA(x...)` and `SA((x...，))`
struct Args{T<:Tuple}
    args::T
end
Length(x::Args) = Length(length(x.args))
const BadArgs = Args{<:Tuple{Tuple{<:Tuple}}}

# Some help functions.
@pure has_ndims(::Type{<:StaticArray{<:Tuple,<:Any,N}}) where {N} = @isdefined N
@pure has_ndims(::Type{<:StaticArray}) = false
if VERSION < v"1.7"
    Base.ndims(::Type{<:StaticArray{<:Tuple,<:Any,N}}) where {N} = N
end
@pure has_eltype(::Type{<:StaticArray{<:Tuple,T}}) where {T} = @isdefined T
@pure has_eltype(::Type{<:StaticArray}) = false
@pure has_size(::Type{<:StaticArray{S}}) where {S<:Tuple} = @isdefined S
@pure has_size(::Type{<:StaticArray}) = false
@pure has_size1(::Type{<:StaticMatrix{M}}) where {M} = @isdefined M
@pure has_size1(::Type{<:StaticMatrix}) = false
_size1(::Type{<:StaticMatrix{M}}) where {M} = M
StaticSquareMatrix{N,T} = StaticMatrix{N,N,T}
@generated function _sqrt(::Length{L}) where {L}
    N = round(Int, sqrt(L))
    N^2 == L || throw(DimensionMismatch("Input's length must be perfect square"))
    return :($N)
end

const FirstClass = Union{SArray, MArray, SHermitianCompact, SizedArray}

"""
    construct_type(::Type{<:StaticArray}, x)

Returns a constructor for a statically-sized array based on `x`'s size and eltype.
The first argument is returned by default.
"""
function construct_type(::Type{SA}, x) where {SA<:StaticArray}
    x isa BadArgs || return SA
    _no_precise_size(SA, x.args[1][1])
end

# Here we define `construct_type(SA, x)` for `SArray`, `MArray`, `SHermitianCompact`, `SizedArray`
# Different `x` has different rules, to summarize:
# 1. Tuple
#    We try to fill `SA` with elements in `x` if `SA` is static-sized.
#    If `SA <: StaticVector`, the output `Length` is derived based on `length(x)`.
#    If `SA <: StaticMatrix{M}`, the output `Size` is derived based on `length(x)÷M`.
#    If `SA <: StaticMatrix{M,M} where M`, the output `Size` is derived based on `sqrt(length(x))`.
#    If `length(SA) == 1 && length（x） > 1`, then we tries to fill `SA` with `x` itself. (rewrapping)
# 2. Args (`SA(x...)`)
#    Similar to `Tuple`, but rewrapping is not allowed.
# 3. StaticArray
#    Treat `x` as `Tuple` whenever possible. If failed, then try to inherit `x`'s `Size`.
# 4. AbstractArray
#    `x` is used to provide eltype. Thus `SA` must be static sized.
function construct_type(::Type{SA}, x) where {SA<:FirstClass}
    SA′ = adapt_eltype_size(SA, x)
    check_parameters(SA′)
    x isa Tuple && SA === SA′ && error("Constructor for $SA is missing. Please file a bug.")
    return SA′
end

adapt_eltype_size(SA, x) = adapt_eltype(adapt_size(SA, x), x)
function adapt_size(::Type{SA}, x) where {SA<:StaticArray}
    if has_size(SA) && length_match_size(SA, x)
        SZ = Tuple{size(SA)...}
    else
        len = x isa Tuple ? length(x) : Int(Length(x))
        len isa Int || _no_precise_size(SA, x)
        if SA <: StaticVector
            SZ = Tuple{len}
        elseif SA <: StaticMatrix && has_size1(SA)
            N = _size1(SA)
            M = len ÷ N
            M * N == len || throw(DimensionMismatch("Incorrect matrix sizes. $len does not divide $N elements"))
            SZ = Tuple{N, M}
        elseif SA <: StaticSquareMatrix
            N = _sqrt(Length(len))
            SZ = Tuple{N, N}
        elseif x isa StaticArray
            SZ = Tuple{size(x)...}
        else
            _no_precise_size(SA, x)
        end
    end
    SA′ = Base.typeintersect(SA, StaticArrayNoEltype{SZ,tuple_length(SZ)})
    SA′ === Union{} && _no_precise_size(SA, x)
    return SA′
end

function length_match_size(::Type{SA}, x) where {SA<:StaticArray}
    if has_ndims(SA)
        SZ, N = size(SA), ndims(SA)
        length(SZ) == N || throw(ArgumentError("Size $(Tuple{SZ...}) mismatches dimension $N."))
    end
    _length_match_size(length(SA), x) || _no_precise_size(SA, x)
end
_length_match_size(l::Int, x::Union{Tuple,StaticArray}) = l == 1 || l == length(x)
_length_match_size(l::Int, x::Args) = l == length(x.args)
_length_match_size(::Int, _) = true

function adapt_eltype(::Type{SA}, x) where {SA<:StaticArray}
    has_eltype(SA) && return SA
    T = if need_rewrap(SA, x)
        typeof(x)
    elseif x isa Tuple
        promote_tuple_eltype(x)
    elseif x isa Args
        promote_tuple_eltype(x.args)
    else
        eltype(x)
    end
    return Base.typeintersect(SA, AbstractArray{T})
end

need_rewrap(::Type{<:StaticArray}, x) = false
function need_rewrap(::Type{SA}, x::Union{Tuple,StaticArray}) where {SA <: StaticArray}
    has_size(SA) && length(SA) == 1 && length(x) > 1
end

check_parameters(::Type{<:FirstClass}) = nothing
check_parameters(::Type{SArray{S,T,N,L}}) where {S<:Tuple,T,N,L} = check_array_parameters(S,T,Val{N},Val{L})
check_parameters(::Type{MArray{S,T,N,L}}) where {S<:Tuple,T,N,L} = check_array_parameters(S,T,Val{N},Val{L})
check_parameters(::Type{SHermitianCompact{N,T,L}}) where {N,T,L} = _check_hermitian_parameters(Val(N), Val(L))

_no_precise_size(SA, x::Args) = _no_precise_size(SA, x.args)
_no_precise_size(SA, x::Tuple) = throw(DimensionMismatch("No precise constructor for $SA found. Length of input was $(length(x))."))
_no_precise_size(SA, x::StaticArray) = throw(DimensionMismatch("No precise constructor for $SA found. Size of input was $(size(x))."))
_no_precise_size(SA, x) = throw(DimensionMismatch("No precise constructor for $SA found. Input is not static sized."))

@inline (::Type{SA})(x...) where {SA <: StaticArray} = construct_type(SA, Args(x))(x)
@inline function (::Type{SA})(x::Tuple) where {SA <: FirstClass}
    SA′ = construct_type(SA, x)
    need_rewrap(SA′, x) ? SA′((x,)) : SA′(x)
end
@inline function (::Type{SA})(sa::StaticArray) where {SA <: StaticArray}
    SA′ = construct_type(SA, sa)
    need_rewrap(SA′, sa) ? SA′((sa,)) : SA′(Tuple(sa))
end
@propagate_inbounds (::Type{SA})(a::AbstractArray) where {SA <: StaticArray} = convert(SA, a)

# this covers most conversions and "statically-sized reshapes"
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:StaticArray} = SA(Tuple(sa))
@inline convert(::Type{SA}, sa::StaticArray) where {SA<:Scalar} = SA((sa[],)) # disambiguation
@inline convert(::Type{SA}, sa::SA) where {SA<:StaticArray} = sa
@inline convert(::Type{SA}, x::Tuple) where {SA<:StaticArray} = SA(x) # convert -> constructor. Hopefully no loops...

# support conversion to AbstractArray
AbstractArray{T}(sa::StaticArray{S,T}) where {S,T} = sa
AbstractArray{T,N}(sa::StaticArray{S,T,N}) where {S,T,N} = sa
AbstractArray{T}(sa::StaticArray{S,U}) where {S,T,U} = similar_type(typeof(sa),T,Size(sa))(sa)
AbstractArray{T,N}(sa::StaticArray{S,U,N}) where {S,T,U,N} = similar_type(typeof(sa),T,Size(sa))(sa)

# Constructing a Tuple from a StaticArray
@inline Tuple(a::StaticArray) = unroll_tuple(a, Length(a))

@noinline function dimension_mismatch_fail(SA::Type, a::AbstractArray)
    throw(DimensionMismatch("expected input array of length $(length(SA)), got length $(length(a))"))
end

@propagate_inbounds function convert(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    @boundscheck if length(a) != length(SA)
        dimension_mismatch_fail(SA, a)
    end

    return _convert(SA, a, Length(SA))
end

@inline _convert(SA, a, l::Length) = SA(unroll_tuple(a, l))
@inline _convert(SA::Type{<:StaticArray{<:Tuple,T}}, a, ::Length{0}) where T = similar_type(SA, T)(())
@inline _convert(SA, a, ::Length{0}) = similar_type(SA, eltype(a))(())

length_val(a::T) where {T <: StaticArrayLike} = length_val(Size(T))
length_val(a::Type{T}) where {T<:StaticArrayLike} = length_val(Size(T))

@generated function unroll_tuple(a::AbstractArray, ::Length{L}) where {L}
    exprs = [:(a[$j]) for j = 1:L]
    quote
        @_inline_meta
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

# `float` and `real` of StaticArray types, analogously to application to scalars (issue 935)
float(::Type{SA}) where SA<:StaticArray{_S,T,_N} where {_S,T,_N} = similar_type(SA, float(T))
real(::Type{SA}) where SA<:StaticArray{_S,T,_N} where {_S,T,_N} = similar_type(SA, real(T))