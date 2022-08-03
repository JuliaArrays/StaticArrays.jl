################
## broadcast! ##
################

using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Style, Broadcasted
using Base.Broadcast: broadcast_shape, _broadcast_getindex, combine_axes
import Base.Broadcast: BroadcastStyle, materialize!, instantiate
import Base.Broadcast: _bcs1  # for SOneTo axis information
using Base.Broadcast: _bcsm
# Add a new BroadcastStyle for StaticArrays, derived from AbstractArrayStyle
# A constructor that changes the style parameter N (array dimension) is also required
struct StaticArrayStyle{N} <: AbstractArrayStyle{N} end
StaticArrayStyle{M}(::Val{N}) where {M,N} = StaticArrayStyle{N}()
BroadcastStyle(::Type{<:StaticArray{<:Tuple, <:Any, N}}) where {N} = StaticArrayStyle{N}()
BroadcastStyle(::Type{<:Transpose{<:Any, <:StaticArray}}) = StaticArrayStyle{2}()
BroadcastStyle(::Type{<:Adjoint{<:Any, <:StaticArray}}) = StaticArrayStyle{2}()
BroadcastStyle(::Type{<:Diagonal{<:Any, <:StaticArray{<:Tuple, <:Any, 1}}}) = StaticArrayStyle{2}()
# Precedence rules
BroadcastStyle(::StaticArrayStyle{M}, ::DefaultArrayStyle{N}) where {M,N} =
    DefaultArrayStyle(Val(max(M, N)))
BroadcastStyle(::StaticArrayStyle{M}, ::DefaultArrayStyle{0}) where {M} =
    StaticArrayStyle{M}()

# combine_axes overload (for Tuple)
@inline static_combine_axes(A, B...) = broadcast_shape(static_axes(A), static_combine_axes(B...))
static_combine_axes(A) = static_axes(A)
static_axes(A) = axes(A)
static_axes(x::Tuple) = (SOneTo{length(x)}(),)
static_axes(bc::Broadcasted{Style{Tuple}}) = static_combine_axes(bc.args...)
Broadcast._axes(bc::Broadcasted{<:StaticArrayStyle}, ::Nothing) = static_combine_axes(bc.args...)

# instantiate overload
@inline function instantiate(B::Broadcasted{StaticArrayStyle{M}}) where M
    if B.axes isa Tuple{Vararg{SOneTo}} || B.axes isa Tuple && length(B.axes) > M
        return invoke(instantiate, Tuple{Broadcasted}, B)
    elseif B.axes isa Nothing
        ax = static_combine_axes(B.args...)
        return Broadcasted{StaticArrayStyle{M}}(B.f, B.args, ax)
    else
        # We need to update B.axes for `broadcast!` if it's not static and `ndims(dest) < M`.
        ax = static_check_broadcast_shape(B.axes, static_combine_axes(B.args...))
        return Broadcasted{StaticArrayStyle{M}}(B.f, B.args, ax)
    end
end
@inline function static_check_broadcast_shape(shp::Tuple, Ashp::Tuple{Vararg{SOneTo}})
    ax1 = if length(Ashp[1]) == 1
        shp[1]
    elseif Ashp[1] == shp[1]
        Ashp[1]
    else
        throw(DimensionMismatch("array could not be broadcast to match destination"))
    end
    return (ax1, static_check_broadcast_shape(Base.tail(shp), Base.tail(Ashp))...)
end
static_check_broadcast_shape(::Tuple{}, ::Tuple{SOneTo,Vararg{SOneTo}}) =
    throw(DimensionMismatch("cannot broadcast array to have fewer non-singleton dimensions"))
static_check_broadcast_shape(::Tuple{}, ::Tuple{SOneTo{1},Vararg{SOneTo{1}}}) = ()
static_check_broadcast_shape(::Tuple{}, ::Tuple{}) = ()
# copy overload
@inline function Base.copy(B::Broadcasted{StaticArrayStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    ax = axes(B)
    ax isa Tuple{Vararg{SOneTo}} || error("Dimension is not static. Please file a bug.")
    return _broadcast(f, Size(map(length, ax)), argsizes, as...)
end
# copyto! overloads
@inline Base.copyto!(dest, B::Broadcasted{<:StaticArrayStyle}) = _copyto!(dest, B)
@inline Base.copyto!(dest::AbstractArray, B::Broadcasted{<:StaticArrayStyle}) = _copyto!(dest, B)
@inline function _copyto!(dest, B::Broadcasted{StaticArrayStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    ax = axes(B)
    if ax isa Tuple{Vararg{SOneTo}}
        @boundscheck axes(dest) == ax || Broadcast.throwdm(axes(dest), ax)
        return _broadcast!(f, Size(map(length, ax)), dest, argsizes, as...)
    end
    # destination dimension cannot be determined statically; fall back to generic broadcast!
    return copyto!(dest, convert(Broadcasted{DefaultArrayStyle{M}}, B))
end

# Resolving priority between dynamic and static axes
_bcs1(a::SOneTo, b::SOneTo) = _bcsm(b, a) ? b : (_bcsm(a, b) ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
function _bcs1(a::SOneTo, b::Base.OneTo)
    length(a) == 1 && return b
    if length(b) != length(a) && length(b) != 1
        throw(DimensionMismatch("arrays could not be broadcast to a common size"))
    end
    return a
end
_bcs1(a::Base.OneTo, b::SOneTo) = _bcs1(b, a)

###################################################
## Internal broadcast machinery for StaticArrays ##
###################################################

# TODO: just use map(broadcast_size, as)?
@inline broadcast_sizes(a, as...) = (broadcast_size(a), broadcast_sizes(as...)...)
@inline broadcast_sizes() = ()
@inline broadcast_size(a) = Size()
@inline broadcast_size(a::AbstractArray) = Size(a)
@inline broadcast_size(a::Tuple) = Size(length(a))

broadcast_getindex(::Tuple{}, i::Int, I::CartesianIndex) = return :(_broadcast_getindex(a[$i], $I))
function broadcast_getindex(oldsize::Tuple, i::Int, newindex::CartesianIndex)
    li = LinearIndices(oldsize)
    ind = _broadcast_getindex(li, newindex)
    return :(a[$i][$ind])
end

isstatic(::StaticArray) = true
isstatic(::Transpose{<:Any, <:StaticArray}) = true
isstatic(::Adjoint{<:Any, <:StaticArray}) = true
isstatic(::Diagonal{<:Any, <:StaticArray}) = true
isstatic(_) = false

@inline first_statictype(x, y...) = isstatic(x) ? typeof(x) : first_statictype(y...)
first_statictype() = error("unresolved dest type")

@inline function _broadcast(f, sz::Size{newsize}, s::Tuple{Vararg{Size}}, a...) where newsize
    first_staticarray = first_statictype(a...)
    if prod(newsize) == 0
        # Use inference to get eltype in empty case (see also comments in _map)
        eltys = Tuple{map(eltype, a)...}
        T = Core.Compiler.return_type(f, eltys)
        @inbounds return similar_type(first_staticarray, T, Size(newsize))()
    end
    elements = __broadcast(f, sz, s, a...)
    @inbounds return similar_type(first_staticarray, eltype(elements), Size(newsize))(elements)
end

@generated function __broadcast(f, ::Size{newsize}, s::Tuple{Vararg{Size}}, a...) where newsize
    sizes = [sz.parameters[1] for sz ∈ s.parameters]

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = (broadcast_getindex(sz, i, current_ind) for (i, sz) in enumerate(sizes))
        exprs[j] = :(f($(exprs_vals...)))
    end

    return quote
        @_inline_meta
        @inbounds return elements = tuple($(exprs...))
    end
end

####################################################
## Internal broadcast! machinery for StaticArrays ##
####################################################

@generated function _broadcast!(f, ::Size{newsize}, dest::AbstractArray, s::Tuple{Vararg{Size}}, a...) where {newsize}
    sizes = [sz.parameters[1] for sz in s.parameters]

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = (broadcast_getindex(sz, i, current_ind) for (i, sz) in enumerate(sizes))
        exprs[j] = :(dest[$j] = f($(exprs_vals...)))
    end

    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end
