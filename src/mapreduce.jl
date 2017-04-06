@inline _first(a1, as...) = a1

################
## map / map! ##
################

# The following type signature for map() matches any list of AbstractArrays,
# provided at least one is a static array.
@inline function map(f, as::Union{SA,AbstractArray}...) where {SA<:StaticArray}
    _map(f, same_size(as...), as...)
end

@generated function _map(f, ::Size{S}, a::AbstractArray...) where {S}
    exprs = Vector{Expr}(prod(S))
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end
    eltypes = [eltype(a[j]) for j ∈ 1:length(a)] # presumably, `eltype` is "hyperpure"?
    newT = :(Core.Inference.return_type(f, Tuple{$(eltypes...)}))
    return quote
        @_inline_meta
        @inbounds return similar_type(typeof(_first(a...)), $newT, Size(S))(tuple($(exprs...)))
    end
end

@inline function map!(f, dest::StaticArray, a::StaticArray...)
    _map!(f, dest, same_size(dest, a...), a...)
end

# Ambiguities with Base:
@inline function map!(f, dest::StaticArray, a::StaticArray)
    _map!(f, dest, same_size(dest, a), a)
end
@inline function map!(f, dest::StaticArray, a::StaticArray, b::StaticArray)
    _map!(f, dest, same_size(dest, a, b), a, b)
end


@generated function _map!(f, dest, ::Size{S}, a::StaticArray...) where {S}
    exprs = Vector{Expr}(prod(S))
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(dest[$i] = f($(tmp...)))
    end
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

###############
## mapreduce ##
###############

@inline function mapreduce(f, op, a::StaticArray, b::StaticArray...)
    _mapreduce(f, op, same_size(a, b...), a, b...)
end

@inline function mapreduce(f, op, v0, a::StaticArray, b::StaticArray...)
    _mapreduce(f, op, v0, same_size(a, b...), a, b...)
end

@generated function _mapreduce(f, op, ::Size{S}, a::StaticArray...) where {S}
    tmp = [:(a[$j][1]) for j ∈ 1:length(a)]
    expr = :(f($(tmp...)))
    for i ∈ 2:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        expr = :(op($expr, f($(tmp...))))
    end
    return quote
        @_inline_meta
        @inbounds return $expr
    end
end

@generated function _mapreduce(f, op, v0, ::Size{S}, a::StaticArray...) where {S}
    expr = :v0
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        expr = :(op($expr, f($(tmp...))))
    end
    return quote
        @_inline_meta
        @inbounds return $expr
    end
end

##################
## mapreducedim ##
##################

# I'm not sure why the signature for this from Base precludes multiple arrays?
# (also, why now mutating `mapreducedim!` and `reducedim!`?)
# (similarly, `broadcastreduce` and `broadcastreducedim` sounds useful)
@inline function mapreducedim(f, op, a::StaticArray, ::Type{Val{D}}) where {D}
    _mapreducedim(f, op, Size(a), a, Val{D})
end

@inline function mapreducedim(f, op, a::StaticArray, ::Type{Val{D}}, v0) where {D}
    _mapreducedim(f, op, Size(a), a, Val{D}, v0)
end

@generated function _mapreducedim(f, op, ::Size{S}, a::StaticArray, ::Type{Val{D}}) where {S, D}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...)

    exprs = Array{Expr}(Snew)
    itr = [1:n for n ∈ Snew]
    for i ∈ Base.product(itr...)
        expr = :(f(a[$(i...)]))
        for k = 2:S[D]
            ik = collect(i)
            ik[D] = k
            expr = :(op($expr, f(a[$(ik...)])))
        end

        exprs[i...] = expr
    end

    # TODO element type might change
    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size($Snew))(tuple($(exprs...)))
    end
end

@generated function _mapreducedim(f, op, ::Size{S}, a::StaticArray, ::Type{Val{D}}, v0) where {S, D}
    N = ndims(a)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...)

    exprs = Array{Expr}(Snew)
    itr = [1:n for n = Snew]
    for i ∈ Base.product(itr...)
        expr = :v0
        for k = 1:S[D]
            ik[D] = k
            expr = :(op($expr, f(a[$(ik...)])))
        end

        exprs[i...] = expr
    end

    # TODO element type might change
    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size($Snew))(tuple($(exprs...)))
    end
end

############
## reduce ##
############

@inline reduce(op, a::StaticArray) = mapreduce(identity, op, a)
@inline reduce(op, v0, a::StaticArray) = mapreduce(identity, op, v0, a)

###############
## reducedim ##
###############

@inline reducedim(op, a::StaticArray, ::Val{D}) where {D} = mapreducedim(identity, op, a, Val{D})
@inline reducedim(op, a::StaticArray, ::Val{D}, v0) where {D} = mapreducedim(identity, op, a, Val{D}, v0)

#######################
## related functions ##
#######################

# These are all similar in Base but not @inline'd
@inline sum(a::StaticArray{T}) where {T} = reduce(+, zero(T), a)
@inline prod(a::StaticArray{T}) where {T} = reduce(*, one(T), a)
@inline count(a::StaticArray{Bool}) = reduce(+, 0, a)
@inline all(a::StaticArray{Bool}) = reduce(&, true, a)  # non-branching versions
@inline any(a::StaticArray{Bool}) = reduce(|, false, a) # (benchmarking needed)
@inline mean(a::StaticArray) = sum(a) / length(a)
@inline sumabs(a::StaticArray{T}) where {T} = mapreduce(abs, +, zero(T), a)
@inline sumabs2(a::StaticArray{T}) where {T} = mapreduce(abs2, +, zero(T), a)
@inline minimum(a::StaticArray) = reduce(min, a) # base has mapreduce(idenity, scalarmin, a)
@inline maximum(a::StaticArray) = reduce(max, a) # base has mapreduce(idenity, scalarmax, a)
@inline minimum(a::StaticArray, dim::Type{Val{D}}) where {D} = reducedim(min, a, dim)
@inline maximum(a::StaticArray, dim::Type{Val{D}}) where {D} = reducedim(max, a, dim)

# Diff is slightly different
@inline diff(a::StaticArray) = diff(a, Val{1})
@inline diff(a::StaticArray, ::Type{Val{D}}) where {D} = _diff(Size(a), a, Val{D})

@generated function _diff(::Size{S}, a::StaticArray, ::Type{Val{D}}) where {S, D}
    N = length(S)
    Snew = ([n==D ? S[n]-1 : S[n] for n = 1:N]...)

    exprs = Array{Expr}(Snew)
    itr = [1:n for n = Snew]

    for i1 = Base.product(itr...)
        i2 = copy([i1...])
        i2[D] = i1[D] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end

    # TODO element type might change
    return quote
        @_inline_meta
        @inbounds return similar_type(a, Size($Snew))(tuple($(exprs...)))
    end
end
