@static if VERSION < v"1.8.0-DEV.410"
    using Base: @_inline_meta
else
    const var"@_inline_meta" = Base.var"@inline"
end

# Julia 1.9 removed the `@pure` annotation in favor of Concrete-Eval
# This behaves unfavorable with `julia --check-bounds=no`
Base.@pure function typeintersect(@nospecialize(a),@nospecialize(b))
    Base.typeintersect(a,b)
end

# For convenience
TupleN{T,N} = NTuple{N,T}

# Cast any Tuple to an TupleN{T}
@inline convert_ntuple(::Type{T},d::T) where {T} = T # For zero-dimensional arrays
@inline convert_ntuple(::Type{T},d::NTuple{N,T}) where {N,T} = d

# Base gives up on tuples for promote_eltype... (TODO can we improve Base?)
_TupleOf{T} = Tuple{T,Vararg{T}}
promote_tuple_eltype(::Union{_TupleOf{T}, Type{<:_TupleOf{T}}}) where {T} = T
@generated function promote_tuple_eltype(::Union{T,Type{T}}) where T <: Tuple
    t = Union{}
    for i = 1:length(T.parameters)
        tmp = Base.unwrapva(T.parameters[i])
        t = :(promote_type($t, $tmp))
    end
    return quote
        @_inline_meta
        $t
    end
end

# Something doesn't match up type wise
function check_array_parameters(Size, T, N, L)
    (!isa(Size, DataType) || (Size.name !== Tuple.name)) && throw(ArgumentError("Static Array parameter Size must be a Tuple type, got $Size"))
    !isa(T, Type) && throw(ArgumentError("Static Array parameter T must be a type, got $T"))
    !isa(N.parameters[1], Int) && throw(ArgumentError("Static Array parameter N must be an integer, got $(N.parameters[1])"))
    !isa(L.parameters[1], Int) && throw(ArgumentError("Static Array parameter L must be an integer, got $(L.parameters[1])"))
    # shouldn't reach here. Anything else should have made it to the function below
    error("Internal error. Please file a bug")
end

# Trivial view used to drop static dimensions to override dispatch
struct TrivialView{A,T,N} <: AbstractArray{T,N}
    a::A
end

size(a::TrivialView) = size(a.a)
getindex(a::TrivialView, inds...) = getindex(a.a, inds...)
setindex!(a::TrivialView, inds...) = (setindex!(a.a, inds...); a)
Base.IndexStyle(::Type{<:TrivialView{A}}) where {A} = IndexStyle(A)

TrivialView(a::AbstractArray{T,N}) where {T,N} = TrivialView{typeof(a),T,N}(a)


# Remove the static dimensions from an array

# This docstring is commented out, because of a bug that happens when building the docs.
# See https://github.com/JuliaLang/julia/issues/23826
# """
#     drop_sdims(a)
#
# Return an `AbstractArray` with the same elements as `a`, but with static
# dimensions removed (ie, not a `StaticArray`).
#
# This is useful if you want to override dispatch to call the `Base` version of
# operations such as `kron` instead of the implementation in `StaticArrays`.
# Normally you shouldn't need to do this, but it can be more efficient for
# certain algorithms where the number of elements of the output is a lot larger
# than the input.
# """
@inline drop_sdims(a::StaticArrayLike) = TrivialView(a)
@inline drop_sdims(a) = a

Base.@propagate_inbounds function invperm(p::StaticVector)
    # in difference to base, this does not check if p is a permutation (every value unique)
     ip = similar(p)
     ip[p] = 1:length(p)
     similar_type(p)(ip)
end

# `ccall` passes a static array as a pointer from a `RefValue` holding it, which
# Base converts to a `Ptr{Cvoid}` or to a pointer to its own type. A pointer to the
# elements is taken through this wrapper. A method on `Base.RefValue` itself would tie
# the element type `T` of `Ptr{T}` to the array in the `RefValue`, and type
# intersection cannot tell such a method apart from Base's
# `unsafe_convert(::Type{Ptr{T}}, ::RefValue{T})` for an abstract `T`: loading
# StaticArrays would invalidate code compiled for Base's method.
struct RefElements{A<:StaticArray}
    ref::Base.RefValue{A}
end
Base.unsafe_convert(::Type{Ptr{T}}, r::RefElements{A}) where {T, A<:StaticArray{<:Tuple, T}} =
    Ptr{T}(Base.unsafe_convert(Ptr{A}, r.ref))

# The argument `ccall` converts to a `Ptr{T}` for `r`: the wrapper for a pointer to
# the elements (see the methods next to each `cconvert`), else `r` for Base to convert.
# `T` is not tied to the element type in the signature of `cconvert`, for the reason
# above.
ref_pointer(::Type, r::Base.RefValue) = r
