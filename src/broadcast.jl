################
## broadcast! ##
################

# TODO: bad codegen for `broadcast(-, SVector(1,2,3))`

@propagate_inbounds function broadcast(f, a::Union{Number, StaticArray}...)
    _broadcast(f, broadcast_sizes(a...), a...)
end

@inline broadcast_sizes(a...) = _broadcast_sizes((), a...)
@inline _broadcast_sizes(t::Tuple) = t
@inline _broadcast_sizes(t::Tuple, a::StaticArray, as...) = _broadcast_sizes((t..., Size(a)), as...)
@inline _broadcast_sizes(t::Tuple, a::Number, as...) = _broadcast_sizes((t..., Size()), as...)

function broadcasted_index(oldsize, newindex)
    index = ones(Int, length(oldsize))
    for i = 1:length(oldsize)
        if oldsize[i] != 1
            index[i] = newindex[i]
        end
    end
    return sub2ind(oldsize, index...)
end

@generated function _broadcast(f, s::Tuple{Vararg{Size}}, a::Union{Number, StaticArray}...)
    first_staticarray = 0
    for i = 1:length(a)
        if a[i] <: StaticArray
            first_staticarray = a[i]
            break
        end
    end

    sizes = [sz.parameters[1] for sz ∈ s.parameters]

    ndims = 0
    for i = 1:length(sizes)
        ndims = max(ndims, length(sizes[i]))
    end

    newsize = ones(Int, ndims)
    for i = 1:length(sizes)
        s = sizes[i]
        for j = 1:length(s)
            if newsize[j] == 1 || newsize[j] == s[j]
                newsize[j] = s[j]
            else
                throw(DimensionMismatch("Tried to broadcast on inputs sized $sizes"))
            end
        end
    end
    newsize = tuple(newsize...)

    exprs = Array{Expr}(newsize)
    more = newsize[1] != 0
    current_ind = ones(Int, length(newsize))

    while more
        exprs_vals = [(a[i] <: Number ? :(a[$i]) : :(a[$i][$(broadcasted_index(sizes[i], current_ind))])) for i = 1:length(sizes)]
        exprs[current_ind...] = :(f($(exprs_vals...)))

        # increment current_ind (maybe use CartesianRange?)
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

    eltype_exprs = [:(eltype($t)) for t ∈ a]
    newtype_expr = :(Core.Inference.return_type(f, Tuple{$(eltype_exprs...)}))

    return quote
        @_inline_meta
        @inbounds return similar_type($first_staticarray, $newtype_expr, Size($newsize))(tuple($(exprs...)))
    end
end


################
## broadcast! ##
################

@propagate_inbounds function broadcast!(f, dest::StaticArray, a::Union{Number, StaticArray}...)
    _broadcast!(f, Size(dest), dest, broadcast_sizes(a...), a...)
end


@generated function _broadcast!(f, ::Size{newsize}, dest::StaticArray, s::Tuple{Vararg{Size}}, a::Union{Number, StaticArray}...) where {newsize}
    sizes = [sz.parameters[1] for sz ∈ s.parameters]
    sizes = tuple(sizes...)

    ndims = 0
    for i = 1:length(sizes)
        ndims = max(ndims, length(sizes[i]))
    end

    for i = 1:length(sizes)
        s = sizes[i]
        for j = 1:length(s)
            if s[j] != 1 && s[j] != (j <= length(newsize) ? newsize[j] : 1)
                throw(DimensionMismatch("Tried to broadcast to destination sized $newsize from inputs sized $sizes"))
            end
        end
    end

    exprs = Array{Expr}(newsize)
    j = 1
    more = newsize[1] != 0
    current_ind = ones(Int, max(length(newsize), length.(sizes)...))
    while more
        exprs_vals = [(a[i] <: Number ? :(a[$i]) : :(a[$i][$(broadcasted_index(sizes[i], current_ind))])) for i = 1:length(sizes)]
        exprs[current_ind...] = :(dest[$j] = f($(exprs_vals...)))

        # increment current_ind (maybe use CartesianRange?)
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
        @inbounds $(Expr(:block, exprs...))
    end
end
