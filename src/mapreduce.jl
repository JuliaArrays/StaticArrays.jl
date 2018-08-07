@inline _first(a1, as...) = a1

################
## map / map! ##
################

# In 0.6 the three methods below could be replaced with
# `map(f, as::Union{<:StaticArray,AbstractArray}...)` which included at least one `StaticArray`
# this is not the case on 0.7 and we instead hope to find a StaticArray in the first two arguments.
@inline function map(f, a1::StaticArray, as::AbstractArray...)
    _map(f, same_size(a1, as...), a1, as...)
end
@inline function map(f, a1::AbstractArray, a2::StaticArray, as::AbstractArray...)
    _map(f, same_size(a1, a2, as...), a1, a2, as...)
end
@inline function map(f, a1::StaticArray, a2::StaticArray, as::AbstractArray...)
    _map(f, same_size(a1, a2, as...), a1, a2, as...)
end

@generated function _map(f, ::Size{S}, a::AbstractArray...) where {S}
    exprs = Vector{Expr}(undef, prod(S))
    for i ∈ 1:prod(S)
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end
    eltypes = [eltype(a[j]) for j ∈ 1:length(a)] # presumably, `eltype` is "hyperpure"?
    newT = :(return_type(f, Tuple{$(eltypes...)}))
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
    exprs = Vector{Expr}(undef, prod(S))
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

@inline function mapreduce(f, op, a::StaticArray, b::StaticArray...; init = nothing)
    _mapreduce(f, op, init, same_size(a, b...), a, b...)
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

_mapreduce(f, op, v0::Nothing, ::Size{S}, a::StaticArray, b::StaticArray...) where {S} = _mapreduce(f, op, Size{S}(), a, b...)

##################
## mapreducedim ##
##################

# I'm not sure why the signature for this from Base precludes multiple arrays?
# (also, why not mutating `mapreducedim!` and `reducedim!`?)
# (similarly, `broadcastreduce` and `broadcastreducedim` sounds useful)
@inline function mapreduce(f, op, a::StaticArray; dims=:, init=nothing)
    _mapreducedim(f, op, Size(a), a, dims, init)
end

@generated function _mapreducedim(f, op, ::Size{S}, a::StaticArray, ::Val{D}) where {S,D}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...,)
    T0 = eltype(a)
    T = :((T1 = return_type(f, Tuple{$T0}); return_type(op, Tuple{T1,T1})))

    exprs = Array{Expr}(undef, Snew)
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

    return quote
        @_inline_meta
        @inbounds return similar_type(a, $T, Size($Snew))(tuple($(exprs...)))
    end
end

@generated function _mapreducedim(f, op, ::Size{S}, a::StaticArray, ::Val{D}, v0::T) where {S,D,T}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...,)

    exprs = Array{Expr}(undef, Snew)
    itr = [1:n for n = Snew]
    for i ∈ Base.product(itr...)
        expr = :v0
        for k = 1:S[D]
            ik = collect(i)
            ik[D] = k
            expr = :(op($expr, f(a[$(ik...)])))
        end

        exprs[i...] = expr
    end

    return quote
        @_inline_meta
        @inbounds return similar_type(a, T, Size($Snew))(tuple($(exprs...)))
    end
end

@inline _mapreducedim(f, op, ::Size{S}, a::StaticArray, dims::Colon, v0::Nothing) where {S} = _mapreduce(f, op, Size{S}(), a::StaticArray)
@inline _mapreducedim(f, op, ::Size{S}, a::StaticArray, dims::Colon, v0) where {S} = _mapreduce(f, op, v0, Size{S}(), a::StaticArray)
@inline _mapreducedim(f, op, ::Size{S}, a::StaticArray, dims::Val{D}, v0::Nothing) where {D,S} = _mapreducedim(f, op, Size{S}(), a::StaticArray, dims)

############
## reduce ##
############

@inline reduce(op, a::StaticArray; init=nothing, dims=:) = mapreduce(identity, op, a, init=init, dims=dims)

#######################
## related functions ##
#######################

# These are all similar in Base but not @inline'd
#
# Implementation notes:
#
# 1. mapreduce usually does not take initial value v0, because we don't
# always know the return type of an arbitrary mapping function f.  (We usually want to use
# some initial value such as one(T) or zero(T) as v0, where T is the return type of f, but
# if users provide type-unstable f, its return type cannot be known.)  Therefore, mapped
# versions of the functions implemented below usually require the collection to have at
# least two entries.
#
# 2. Exceptions are the ones that require Boolean mapping functions.  For example, f in
# all and any must return Bool, so we know the appropriate v0 is true and false,
# respectively.  Therefore, all(f, ...) and any(f, ...) are implemented by mapreduce(f, ...)
# with an initial value true and false.
@inline iszero(a::StaticArray{<:Any,T}) where {T} = reduce((x,y) -> x && (y==zero(T)), a, init=true)

@inline sum(a::StaticArray{<:Any,T}; dims=:) where {T} = reduce(+, a, dims=dims, init=zero(T))
@inline sum(f::Function, a::StaticArray; dims=:) = mapreduce(f, +, a, dims=dims)

@inline prod(a::StaticArray{<:Any,T}; dims=:) where {T} = reduce(*, a, dims=dims, init=one(T))
@inline prod(f::Function, a::StaticArray{<:Any,T}; dims=:) where {T} = mapreduce(f, *, a, dims=dims)

@inline count(a::StaticArray{<:Any,Bool}; dims=:) = reduce(+, a, dims=dims, init=0)
@inline count(f::Function, a::StaticArray; dims=:) = mapreduce(x->f(x)::Bool, +, a, dims=dims, init=0)

@inline all(a::StaticArray{<:Any,Bool}; dims=:) = reduce(&, a, dims=dims, init=true)
@inline all(f::Function, a::StaticArray; dims=:) = mapreduce(x->f(x)::Bool, &, a, dims=dims, init=true)

@inline any(a::StaticArray{<:Any,Bool}; dims=:) = reduce(|, a, dims=dims, init=false)
@inline any(f::Function, a::StaticArray; dims=:) = mapreduce(x->f(x)::Bool, |, a, dims=dims, init=false)

@inline mean(a::StaticArray; dims=:) = sum(a, dims=dims) / size(a, dims)
@inline mean(f::Function, a::StaticArray; dims=:) = sum(f, a, dims=dims) / size(a, dims)

@inline minimum(a::StaticArray; dims=:) = reduce(min, a, dims=dims)
@inline minimum(f::Function, a::StaticArray; dims=:) = mapreduce(f, min, a, dims=dims)

@inline maximum(a::StaticArray; dims=:) = reduce(max, a, dims=dims)
@inline maximum(f::Function, a::StaticArray; dims=:) = mapreduce(f, max, a, dims=dims)

# Diff is slightly different
@inline diff(a::StaticArray; dims=Val(1)) = _diff(Size(a), a, dims)

@generated function _diff(::Size{S}, a::StaticArray, ::Val{D}) where {S,D}
    N = length(S)
    Snew = ([n==D ? S[n]-1 : S[n] for n = 1:N]...,)

    exprs = Array{Expr}(undef, Snew)
    itr = [1:n for n = Snew]

    for i1 = Base.product(itr...)
        i2 = copy([i1...])
        i2[D] = i1[D] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end

    return quote
        @_inline_meta
        T = typeof(one(eltype(a)) - one(eltype(a)))
        @inbounds return similar_type(a, T, Size($Snew))(tuple($(exprs...)))
    end
end
