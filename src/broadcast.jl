################
## broadcast! ##
################

using Base.Broadcast: AbstractArrayStyle, DefaultArrayStyle, Style, Broadcasted
using Base.Broadcast: broadcast_shape, _broadcast_getindex, combine_axes
import Base.Broadcast: BroadcastStyle, materialize!, instantiate
import Base.Broadcast: _bcs1  # for SOneTo axis information
using Base.Broadcast: _bcsm

BroadcastStyle(::Type{<:StaticArray{<:Tuple, <:Any, N}}) where {N} = StaticArrayStyle{N}()
BroadcastStyle(::Type{<:StaticMatrixLike}) = StaticArrayStyle{2}()
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
    flat = broadcast_flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    ax = axes(B)
    ax isa Tuple{Vararg{SOneTo}} || error("Dimension is not static. Please file a bug.")
    return _broadcast(f, Size(map(length, ax)), argsizes, as...)
end
# copyto! overloads
@inline Base.copyto!(dest::AbstractArray, B::Broadcasted{<:StaticArrayStyle}) = _copyto!(dest, B)
@inline function _copyto!(dest, B::Broadcasted{StaticArrayStyle{M}}) where M
    flat = broadcast_flatten(B); as = flat.args; f = flat.f
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

isstatic(::StaticArrayLike) = true
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
        return tuple($(exprs...))
    end
end

####################################################
## Internal broadcast! machinery for StaticArrays ##
####################################################

@generated function _broadcast!(f, ::Size{newsize}, dest::AbstractArray, s::Tuple{Vararg{Size}}, a...) where {newsize}
    sizes = [sz.parameters[1] for sz in s.parameters]

    indices = CartesianIndices(newsize)
    exprs_eval = similar(indices, Expr)
    exprs_setindex = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = (broadcast_getindex(sz, i, current_ind) for (i, sz) in enumerate(sizes))
        symb_val_j = Symbol(:val_, j)
        exprs_eval[j] = :($symb_val_j = f($(exprs_vals...)))
        exprs_setindex[j] = :(dest[$j] = $symb_val_j)
    end

    return quote
        @_inline_meta
        $(Expr(:block, exprs_eval...))
        @inbounds $(Expr(:block, exprs_setindex...))
        return dest
    end
end

# Work around for https://github.com/JuliaLang/julia/issues/27988
# The following code is borrowed from https://github.com/JuliaLang/julia/pull/43322
# with some modification to make it also works on 1.6.
module StableFlatten

export broadcast_flatten

if VERSION >= v"1.11.0-DEV.103"
    const broadcast_flatten = Broadcast.flatten
else
    using Base: tail
    using Base.Broadcast: isflat, Broadcasted

    maybeconstructor(f) = f
    maybeconstructor(::Type{F}) where {F} = (args...; kwargs...) -> F(args...; kwargs...)

    function broadcast_flatten(bc::Broadcasted{Style}) where {Style}
        isflat(bc) && return bc
        args = cat_nested(bc)
        len = Val{length(args)}()
        makeargs = make_makeargs(bc.args, len, ntuple(_->true, len))
        f = maybeconstructor(bc.f)
        @inline newf(args...) = f(prepare_args(makeargs, args)...)
        return Broadcasted{Style}(newf, args, bc.axes)
    end

    cat_nested(bc::Broadcasted) = cat_nested_args(bc.args)
    cat_nested_args(::Tuple{}) = ()
    cat_nested_args(t::Tuple) = (cat_nested(t[1])..., cat_nested_args(tail(t))...)
    cat_nested(@nospecialize(a)) = (a,)

    function make_makeargs(args::Tuple, len, flags)
        makeargs, r = _make_makeargs(args, len, flags)
        r isa Tuple{} || error("Internal error. Please file a bug")
        return makeargs
    end

    # We build `makeargs` by traversing the broadcast nodes recursively.
    # note: `len` isa `Val` indicates the length of whole flattened argument list.
    #       `flags` is a tuple of `Bool` with the same length of the rest arguments.
    @inline function _make_makeargs(args::Tuple, len::Val, flags::Tuple)
        head, flags′ = _make_makeargs1(args[1], len, flags)
        rest, flags″ = _make_makeargs(tail(args), len, flags′)
        (head, rest...), flags″
    end
    _make_makeargs(::Tuple{}, ::Val, x::Tuple) = (), x

    # For flat nodes:
    # 1. we just consume one argument, and return the "pick" function
    @inline function _make_makeargs1(@nospecialize(a), ::Val{N}, flags::Tuple) where {N}
        pickargs(::Val{N}) where {N} = (@nospecialize(x::Tuple)) -> x[N]
        return pickargs(Val{N-length(flags)+1}()), tail(flags)
    end

    # For nested nodes, we form the `makeargs1` based on the child `makeargs` (n += length(cat_nested(bc)))
    @inline function _make_makeargs1(bc::Broadcasted, len::Val, flags::Tuple)
        makeargs, flags′ = _make_makeargs(bc.args, len, flags)
        f = maybeconstructor(bc.f)
        @inline makeargs1(@nospecialize(args::Tuple)) = f(prepare_args(makeargs, args)...)
        makeargs1, flags′
    end

    prepare_args(::Tuple{}, @nospecialize(::Tuple)) = ()
    @inline prepare_args(makeargs::Tuple, @nospecialize(x::Tuple)) = (makeargs[1](x), prepare_args(tail(makeargs), x)...)
end
end
using .StableFlatten
