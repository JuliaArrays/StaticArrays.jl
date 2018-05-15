################
## broadcast! ##
################

@static if VERSION < v"0.7.0-DEV.5096"
    ## Old Broadcast API ##
    import Base.Broadcast:
    _containertype, promote_containertype, broadcast_indices, containertype,
    broadcast_c, broadcast_c!

    # Add StaticArray as a new output type in Base.Broadcast promotion machinery.
    # This isn't the precise output type, just a placeholder to return from
    # promote_containertype, which will control dispatch to our broadcast_c.
    _containertype(::Type{<:StaticArray}) = StaticArray
    _containertype(::Type{<:Adjoint{<:Any,<:StaticVector}}) = StaticArray

    # issue #382; prevent infinite recursion in generic broadcast code:
    Base.Broadcast.broadcast_indices(::Type{StaticArray}, A) = indices(A)

    # With the above, the default promote_containertype gives reasonable defaults:
    #   StaticArray, StaticArray -> StaticArray
    #   Array, StaticArray       -> Array
    #
    # We could be more precise about the latter, but this isn't really possible
    # without using Array{N} rather than Array in Base's promote_containertype.
    #
    # Base also has broadcast with tuple + Array, but while implementing this would
    # be consistent with Base, it's not exactly clear it's a good idea when you can
    # just use an SVector instead?
    promote_containertype(::Type{StaticArray}, ::Type{Any}) = StaticArray
    promote_containertype(::Type{Any}, ::Type{StaticArray}) = StaticArray

    # Override for when output type is deduced to be a StaticArray.
    @inline function broadcast_c(f, ::Type{StaticArray}, as...)
        argsizes = broadcast_sizes(as...)
        destsize = combine_sizes(argsizes)
        _broadcast(f, destsize, argsizes, as...)
    end

    @inline function broadcast_c!(f, ::Type, ::Type{StaticArray}, dest, as...)
        argsizes = broadcast_sizes(as...)
        destsize = combine_sizes((Size(dest), argsizes...))
        Length(destsize) === Length{Dynamic()}() && return broadcast_c!(f, containertype(dest), Array, dest, as...)
        _broadcast!(f, destsize, dest, argsizes, as...)
    end
else
    ## New Broadcast API ##
    import Base.Broadcast:
    BroadcastStyle, AbstractArrayStyle, Broadcasted

    # Add a new BroadcastStyle for StaticArrays, derived from AbstractArrayStyle
    # A constructor that changes the style parameter N (array dimension) is also required
    struct StaticArrayStyle{N} <: AbstractArrayStyle{N} end
    StaticArrayStyle{M}(::Val{N}) where {M,N} = StaticArrayStyle{N}()

    BroadcastStyle(::Type{<:StaticArray{<:Any, <:Any, N}}) where {N} = StaticArrayStyle{N}()
    BroadcastStyle(::Type{<:Adjoint{<:Any, <:StaticVector}}) = StaticArrayStyle{2}()
    BroadcastStyle(::Type{<:Adjoint{<:Any, <:StaticMatrix}}) = StaticArrayStyle{2}()

    # Precedence rules
    BroadcastStyle(::StaticArrayStyle{M}, ::Broadcast.DefaultArrayStyle{N}) where {M,N} =
        Broadcast.DefaultArrayStyle(Broadcast._max(Val(M), Val(N)))
    BroadcastStyle(::StaticArrayStyle{M}, ::Broadcast.DefaultArrayStyle{0}) where {M} =
        StaticArrayStyle{M}()

    # Add a specialized broadcast method that overrides the Base fallback
    @inline function Base.copy(B::Broadcasted{StaticArrayStyle{M}}) where M
        flat = Broadcast.flatten(B); as = flat.args; f = flat.f
        argsizes = broadcast_sizes(as...)
        destsize = combine_sizes(argsizes)
        # TODO: use the following to fall back to generic broadcast once the precedence rules are less conservative:
        # Length(destsize) === Length{Dynamic()}() && return broadcast(f, Broadcast.DefaultArrayStyle{M}(), nothing, nothing, as...)
        _broadcast(f, destsize, argsizes, as...)
    end

    # Add a specialized broadcast! method that overrides the Base fallback
    @inline function Base.copyto!(dest, B::Broadcasted{StaticArrayStyle{M}}) where M
        flat = Broadcast.flatten(B); as = flat.args; f = flat.f
        argsizes = broadcast_sizes(as...)
        destsize = combine_sizes((Size(dest), argsizes...))
        Length(destsize) === Length{Dynamic()}() && error("Dynamic not implemented")#return broadcast!(f, dest, Broadcast.DefaultArrayStyle{M}(), as...)
        _broadcast!(f, destsize, dest, argsizes, as...)
    end
end


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
        exprs_vals = [(!(a[i] <: AbstractArray) ? :(a[$i][]) : :(a[$i][$(broadcasted_index(sizes[i], current_ind))])) for i = 1:length(sizes)]
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

    eltype_exprs = [t <: Union{AbstractArray, Ref} ? :(eltype($t)) : :($t) for t ∈ a]
    newtype_expr = :(return_type(f, Tuple{$(eltype_exprs...)}))

    return quote
        @_inline_meta
        @inbounds return similar_type($first_staticarray, $newtype_expr, Size(newsize))(tuple($(exprs...)))
    end
end

if VERSION < v"0.7.0-DEV"
# Workaround for #329
    @inline function Base.broadcast(f, ::Type{T}, a::StaticArray) where {T}
        map(x->f(T,x), a)
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
        @_inline_meta
        @boundscheck sizematch($(Size{newsize}()), dest) || throw(DimensionMismatch("array could not be broadcast to match destination"))
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end
