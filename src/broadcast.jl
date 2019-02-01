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
scalar_getindex(x::Tuple{<: Any}) = x[1]

@generated function _broadcast(f, ::Size{newsize}, s::Tuple{Vararg{Size}}, a...) where newsize
    first_staticarray = 0
    for i = 1:length(a)
        if a[i] <: StaticArray
            first_staticarray = a[i]
            break
        end
    end

    exprs = Array{Expr}(undef, newsize)
    more = prod(newsize) > 0
    current_ind = ones(Int, length(newsize))
    sizes = [sz.parameters[1] for sz ∈ s.parameters]

    while more
        exprs_vals = [(!(a[i] <: AbstractArray) ? :(scalar_getindex(a[$i])) : :(a[$i][$(broadcasted_index(sizes[i], current_ind))])) for i = 1:length(sizes)]
        exprs[current_ind...] = :(f($(exprs_vals...)))

        # increment current_ind (maybe use CartesianIndices?)
        current_ind[1] += 1
        for i ∈ 1:length(newsize)
            if current_ind[i] > newsize[i]
                if i == length(newsize)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
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

    ndims = 0
    for i = 1:length(sizes)
        ndims = max(ndims, length(sizes[i]))
    end

    exprs = Array{Expr}(undef, newsize)
    j = 1
    more = prod(newsize) > 0
    current_ind = ones(Int, max(length(newsize), length.(sizes)...))
    while more
        exprs_vals = [(!(as[i] <: AbstractArray) ? :(as[$i][]) : :(as[$i][$(broadcasted_index(sizes[i], current_ind))])) for i = 1:length(sizes)]
        exprs[current_ind...] = :(dest[$j] = f($(exprs_vals...)))

        # increment current_ind (maybe use CartesianIndices?)
        current_ind[1] += 1
        for i ∈ 1:length(newsize)
            if current_ind[i] > newsize[i]
                if i == length(newsize)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
        j += 1
    end

    return quote
        @_propagate_inbounds_meta
        @boundscheck sizematch($(Size{newsize}()), dest) || throw(DimensionMismatch("array could not be broadcast to match destination"))
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end
