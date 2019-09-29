################
## broadcast! ##
################

import Base.Broadcast:
BroadcastStyle, AbstractArrayStyle, Broadcasted, DefaultArrayStyle, materialize!
import Base.Broadcast: _bcs1  # for SOneTo axis information
using Base.Broadcast: _bcsm
# Add a new BroadcastStyle for StaticArrays, derived from AbstractArrayStyle
# A constructor that changes the style parameter N (array dimension) is also required
struct StaticArrayStyle{N} <: AbstractArrayStyle{N} end
StaticArrayStyle{M}(::Val{N}) where {M,N} = StaticArrayStyle{N}()
BroadcastStyle(::Type{<:StaticArray{<:Tuple, <:Any, N}}) where {N} = StaticArrayStyle{N}()
BroadcastStyle(::Type{<:Transpose{<:Any, <:StaticArray{<:Tuple, <:Any, N}}}) where {N} = StaticArrayStyle{N}()
BroadcastStyle(::Type{<:Adjoint{<:Any, <:StaticArray{<:Tuple, <:Any, N}}}) where {N} = StaticArrayStyle{N}()
# Precedence rules
BroadcastStyle(::StaticArrayStyle{M}, ::DefaultArrayStyle{N}) where {M,N} =
    DefaultArrayStyle(Val(max(M, N)))
BroadcastStyle(::StaticArrayStyle{M}, ::DefaultArrayStyle{0}) where {M} =
    StaticArrayStyle{M}()
# copy overload
@inline function Base.copy(B::Broadcasted{StaticArrayStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    destsize = combine_sizes(argsizes)
    _broadcast(f, destsize, argsizes, as...)
end
# copyto! overloads
@inline Base.copyto!(dest, B::Broadcasted{<:StaticArrayStyle}) = _copyto!(dest, B)
@inline Base.copyto!(dest::AbstractArray, B::Broadcasted{<:StaticArrayStyle}) = _copyto!(dest, B)
@inline function _copyto!(dest, B::Broadcasted{StaticArrayStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = broadcast_sizes(as...)
    destsize = combine_sizes((Size(dest), argsizes...))
    if Length(destsize) === Length{Dynamic()}()
        # destination dimension cannot be determined statically; fall back to generic broadcast!
        return copyto!(dest, convert(Broadcasted{DefaultArrayStyle{M}}, B))
    end
    _broadcast!(f, destsize, dest, argsizes, as...)
end

# Resolving priority between dynamic and static axes
_bcs1(a::SOneTo, b::SOneTo) = _bcsm(b, a) ? b : (_bcsm(a, b) ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
_bcs1(a::SOneTo, b::Base.OneTo) = _bcs1(Base.OneTo(a), b)
_bcs1(a::Base.OneTo, b::SOneTo) = _bcs1(a, Base.OneTo(b))

###################################################
## Internal broadcast machinery for StaticArrays ##
###################################################

broadcast_indices(A::StaticArray) = indices(A)

# TODO: just use map(broadcast_size, as)?
@inline broadcast_sizes(a, as...) = (broadcast_size(a), broadcast_sizes(as...)...)
@inline broadcast_sizes() = ()
@inline broadcast_size(a) = Size()
@inline broadcast_size(a::AbstractArray) = Size(a)
@inline broadcast_size(a::NTuple{N}) where N = Size(N)

function broadcasted_index(oldsize, newindex)
    index = ones(Int, length(oldsize))
    for i = 1:length(oldsize)
        if oldsize[i] != 1
            index[i] = newindex[i]
        end
    end
    return LinearIndices(oldsize)[index...]
end

# similar to Base.Broadcast.combine_indices:
@generated function combine_sizes(s::Tuple{Vararg{Size}})
    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    ndims = 0
    for i = 1:length(sizes)
        ndims = max(ndims, length(sizes[i]))
    end
    newsize = StaticDimension[Dynamic() for _ = 1 : ndims]
    for i = 1:length(sizes)
        s = sizes[i]
        for j = 1:length(s)
            if s[j] isa Dynamic
                continue
            elseif newsize[j] isa Dynamic || newsize[j] == 1
                newsize[j] = s[j]
            elseif newsize[j] ≠ s[j] && s[j] ≠ 1
                throw(DimensionMismatch("Tried to broadcast on inputs sized $sizes"))
            end
        end
    end
    quote
        @_inline_meta
        Size($(tuple(newsize...)))
    end
end

scalar_getindex(x) = x
scalar_getindex(x::Ref) = x[]

@generated function _broadcast(f, ::Size{newsize}, s::Tuple{Vararg{Size}}, a...) where newsize
    first_staticarray = a[findfirst(ai -> ai <: StaticArray, a)]

    if prod(newsize) == 0
        # Use inference to get eltype in empty case (see also comments in _map)
        eltys = [:(eltype(a[$i])) for i ∈ 1:length(a)]
        return quote
            @_inline_meta
            T = Core.Compiler.return_type(f, Tuple{$(eltys...)})
            @inbounds return similar_type($first_staticarray, T, Size(newsize))()
        end
    end

    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = [
            (!(a[i] <: AbstractArray || a[i] <: Tuple) ? :(scalar_getindex(a[$i])) : :(a[$i][$(broadcasted_index(sizes[i], current_ind))]))
            for i = 1:length(sizes)
        ]
        exprs[j] = :(f($(exprs_vals...)))
    end

    return quote
        @_inline_meta
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type($first_staticarray, eltype(elements), Size(newsize))(elements)
    end
end

####################################################
## Internal broadcast! machinery for StaticArrays ##
####################################################

@generated function _broadcast!(f, ::Size{newsize}, dest::AbstractArray, s::Tuple{Vararg{Size}}, as...) where {newsize}
    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    sizes = tuple(sizes...)

    # TODO: this could also be done outside the generated function:
    sizematch(Size{newsize}(), Size(dest)) ||
        throw(DimensionMismatch("Tried to broadcast to destination sized $newsize from inputs sized $sizes"))

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    for (j, current_ind) ∈ enumerate(indices)
        exprs_vals = [
            (!(as[i] <: AbstractArray || as[i] <: Tuple) ? :(as[$i][]) : :(as[$i][$(broadcasted_index(sizes[i], current_ind))]))
            for i = 1:length(sizes)
        ]
        exprs[j] = :(dest[$j] = f($(exprs_vals...)))
    end

    return quote
        @_propagate_inbounds_meta
        @boundscheck sizematch($(Size{newsize}()), dest) || throw(DimensionMismatch("array could not be broadcast to match destination"))
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end
