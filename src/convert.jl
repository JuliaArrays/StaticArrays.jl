"""
    Args

A help wrapper to distinguish `SA(x...)` and `SA((x...,))`
"""
struct Args{T<:Tuple}
    args::T
end
Length(x::Args) = Length(length(x.args))
const BadArgs = Args{<:Tuple{Tuple{<:Tuple}}}

# Some help functions.
@pure has_ndims(::Type{<:StaticArray{<:Tuple,<:Any,N}}) where {N} = @isdefined N
has_ndims(::Type{<:StaticArray}) = false
if VERSION < v"1.7"
    Base.ndims(::Type{<:StaticArray{<:Tuple,<:Any,N}}) where {N} = N
end
@pure has_eltype(::Type{<:StaticArray{<:Tuple,T}}) where {T} = @isdefined T
has_eltype(::Type{<:StaticArray}) = false
@pure has_size(::Type{<:StaticArray{S}}) where {S<:Tuple} = @isdefined S
has_size(::Type{<:StaticArray}) = false
# workaround for https://github.com/JuliaArrays/StaticArrays.jl/issues/1047
has_size(::Type{SVector}) = false
has_size(::Type{MVector}) = false
has_size(::Type{SMatrix}) = false
has_size(::Type{MMatrix}) = false
has_size(::Type{SMatrix{N}}) where {N} = false
has_size(::Type{MMatrix{N}}) where {N} = false

@pure has_size1(::Type{<:StaticMatrix{M}}) where {M} = @isdefined M
has_size1(::Type{<:StaticMatrix}) = false
_size1(::Type{<:StaticMatrix{M}}) where {M} = M
@generated function _sqrt(::Length{L}) where {L}
    N = isqrt(L)
    N^2 == L && return :($N)
    throw(DimensionMismatch("Input's length must be perfect square"))
end

"""
    SA′ = construct_type(::Type{SA}, x) where {SA<:StaticArray}

Pick a proper constructor `SA′` based on `x` if `SA(x)`/`SA(x...)` has no specific definition.
The default returned `SA′` is `SA` itself for user defined `StaticArray`s. This differs from 
`similar_type()` in that `SA′` should always be a subtype of `SA`.

!!! note
    To distinguish `SA(x...)` and `SA(x::Tuple)`, the former calls 
    `construct_type(SA, StaticArrays.Args(x))` instead of `construct_type(SA, x)`.

!!! note
    Please make sure `SA'(x)` has a specific definition if the default behavior is overloaded. 
    Otherwise construction might fall into infinite recursion.

---
The adaption rules for official `StaticArray`s could be summarized as:

# `SA <: FieldArray`: `eltype` adaptable

`FieldArray`s are always static-sized. We only derive `SA′`'s `eltype` using type promotion if needed.

# `SA <: Union{SArray, MArray, SHermitianCompact, SizedArray}`: `size`/`eltype` adaptable

- `SA(x::Tuple)`

  If `SA` is fully static-sized, then we first try to fill `SA` with `x`'s elements.
  If failed and `length(SA) == 1`, then we try to fill `SA` with `x` itself.

  If `SA` is not fully static-sized, then we always try to fill `SA` with `x`'s elements,
  and the constructor's `Size` is derived based on:
  1. If `SA <: StaticVector`, then we use `length(x)` as the output `Length`
  2. If `SA <: StaticMatrix{M}`, then we use `(M, N)` (`N = length(x) ÷ M`) as the output `Size`
  3. If `SA <: StaticMatrix{M,M} where M`, then we use `(N, N)` (`N = sqrt(length(x)`) as the output `Size`.

- `SA(x...)`

  Similar to `Tuple`, but we never fill `SA` with `x` itself.

- `SA(x::StaticArray)`

  We treat `x` as `Tuple` whenever possible. If failed, then try to inherit `x`'s `Size`.

- `SA(x::AbstractArray)`

  `x` is used to provide eltype. Thus `SA` must be static sized.

"""
function construct_type(::Type{SA}, x) where {SA<:StaticArray}
    x isa BadArgs || return SA
    _no_precise_size(SA, x.args[1][1])
end

# These StaticArrays support `size`/`eltype` adaption during construction.
const SizeEltypeAdaptable = Union{SArray, MArray, SHermitianCompact, SizedArray}
function construct_type(::Type{SA}, x) where {SA<:SizeEltypeAdaptable}
    SA′ = adapt_eltype(adapt_size(SA, x), x)
    check_parameters(SA′)
    (!need_rewrap(SA′, x) && x isa Tuple && SA === SA′) || return SA′
    error("Constructor for $SA is missing. Please file a bug.")
end

adapt_size_impl1(::Type{SA}, ::Any, y::Int) where {SA<:StaticArray} = y
adapt_size_impl1(::Type{SA}, x, ::Any) where {SA<:StaticArray} = _no_precise_size(SA, x)

adapt_size_impl0(::Type{SA}, x, ::Length{y}) where {SA<:StaticArray, y} = adapt_size_impl1(SA, x, y)

function adapt_size(::Type{SA}, x) where {SA<:StaticArray}
    if has_size(SA) && length_match_size(SA, x)
        SZ = Tuple{size(SA)...}
    else
        len = x isa Tuple ? length(x) : adapt_size_impl0(SA, x, Length(x))
        len isa Int || _no_precise_size(SA, x)
        if SA <: StaticVector
            SZ = Tuple{len}
        elseif SA <: StaticMatrix && has_size1(SA)
            N = _size1(SA)
            M = len ÷ N
            M * N == len || throw(DimensionMismatch("Incorrect matrix sizes. $len does not divide $N elements"))
            SZ = Tuple{N, M}
        elseif SA <: StaticMatrix{N,N} where {N}
            N = _sqrt(Length(len))
            SZ = Tuple{N, N}
        elseif x isa StaticArray
            SZ = Tuple{size(x)...}
        else
            _no_precise_size(SA, x)
        end
    end
    SA′ = typeintersect(SA, StaticArrayNoEltype{SZ,tuple_length(SZ)})
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
    return typeintersect(SA, AbstractArray{T})
end

need_rewrap(::Type{<:StaticArray}, x) = false
function need_rewrap(::Type{SA}, x::Union{Tuple,StaticArray}) where {SA <: StaticArray}
    has_size(SA) && length(SA) == 1 && length(x) != 1
end

check_parameters(::Type{<:SizeEltypeAdaptable}) = nothing
check_parameters(::Type{SArray{S,T,N,L}}) where {S<:Tuple,T,N,L} = check_array_parameters(S,T,Val{N},Val{L})
check_parameters(::Type{MArray{S,T,N,L}}) where {S<:Tuple,T,N,L} = check_array_parameters(S,T,Val{N},Val{L})
check_parameters(::Type{SHermitianCompact{N,T,L}}) where {N,T,L} = _check_hermitian_parameters(Val(N), Val(L))

_no_precise_size(SA, x::Args) = _no_precise_size(SA, x.args)
_no_precise_size(SA, x::Tuple) = throw(DimensionMismatch("No precise constructor for $SA found. Length of input was $(length(x))."))
_no_precise_size(SA, x::StaticArray) = throw(DimensionMismatch("No precise constructor for $SA found. Size of input was $(size(x))."))
_no_precise_size(SA, x) = throw(DimensionMismatch("No precise constructor for $SA found. Input is not static sized."))

@inline (::Type{SA})(x...) where {SA <: StaticArray} = construct_type(SA, Args(x))(x)
@inline function (::Type{SA})(x::Tuple) where {SA <: SizeEltypeAdaptable}
    SA′ = construct_type(SA, x)
    need_rewrap(SA′, x) ? SA′((x,)) : SA′(x)
end
@inline function (::Type{SA})(sa::StaticArray) where {SA <: StaticArray}
    SA′ = construct_type(SA, sa)
    need_rewrap(SA′, sa) ? SA′((sa,)) : SA′(Tuple(sa))
end
@propagate_inbounds (T::Type{<:StaticArray})(a::AbstractArray) = convert(T, a)

# this covers most conversions and "statically-sized reshapes"
@inline function convert(::Type{SA}, sa::StaticArray{S}) where {SA<:StaticArray,S<:Tuple}
    SA′ = construct_type(SA, sa)
    # `SA′((sa,))` is not valid. As we want `SA′(sa...)`
    need_rewrap(SA′, sa) && _no_precise_size(SA, sa)
    return SA′(Tuple(sa))
end
@inline convert(::Type{SA}, sa::SA) where {SA<:StaticArray} = sa
@inline convert(::Type{SA}, x::Tuple) where {SA<:StaticArray} = SA(x) # convert -> constructor. Hopefully no loops...

# support conversion to AbstractArray
AbstractArray{T}(sa::StaticArray{S,T}) where {S,T} = sa
AbstractArray{T,N}(sa::StaticArray{S,T,N}) where {S,T,N} = sa
AbstractArray{T}(sa::StaticArray{S,U}) where {S,T,U} = similar_type(typeof(sa),T,Size(sa))(sa)
AbstractArray{T,N}(sa::StaticArray{S,U,N}) where {S,T,U,N} = similar_type(typeof(sa),T,Size(sa))(sa)

# Constructing a Tuple from a StaticArray
@inline Base.Tuple(a::StaticArray) = unroll_tuple(a, Length(a))

@noinline function dimension_mismatch_fail(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    throw(DimensionMismatch("expected input array of length $(length(SA)), got length $(length(a))"))
end

@propagate_inbounds function convert(::Type{SA}, a::AbstractArray) where {SA <: StaticArray}
    @boundscheck if length(a) != length(SA)
        dimension_mismatch_fail(SA, a)
    end
    SA′ = construct_type(SA, a)
    return SA′(unroll_tuple(a, Length(SA′)))
end

length_val(a::T) where {T <: StaticArrayLike} = length_val(Size(T))
length_val(a::Type{T}) where {T<:StaticArrayLike} = length_val(Size(T))

unroll_tuple(a::AbstractArray, ::Length{0}) = ()
unroll_tuple(a::AbstractArray, ::Length{1}) = @inbounds (a[],)
@generated function unroll_tuple(a::AbstractArray, ::Length{L}) where {L}
    exprs = (:(a[$j+Δj]) for j = 0:L-1)
    quote
        @_inline_meta
        Δj = firstindex(a)
        @inbounds return $(Expr(:tuple, exprs...))
    end
end

# `float` and `real` of StaticArray types, analogously to application to scalars (issue 935)
float(::Type{SA}) where SA<:StaticArray{_S,T,_N} where {_S,T,_N} = similar_type(SA, float(T))
real(::Type{SA}) where SA<:StaticArray{_S,T,_N} where {_S,T,_N} = similar_type(SA, real(T))
