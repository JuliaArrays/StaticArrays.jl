@inline _first(a1, as...) = a1

################
## map / map! ##
################

# The following type signature for map() matches any list of AbstractArrays,
# provided at least one is a static array.
if VERSION < v"0.7.0-"
    @inline function map(f, as::Union{SA,AbstractArray}...) where {SA<:StaticArray}
        _map(f, same_size(as...), as...)
    end
else
    @inline function map(f, a1::StaticArray, as::AbstractArray...)
        _map(f, same_size(a1, as...), a1, as...)
    end
end

@generated function _map(f, ::Size{S}, a::AbstractArray...) where {S}
    @compat exprs = Vector{Expr}(uninitialized, prod(S))
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

@generated function _mapreducedim(f, op, ::Size{S}, a::StaticArray, ::Type{Val{D}}) where {S,D}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...)
    T0 = eltype(a)
    T = :((T1 = Core.Inference.return_type(f, Tuple{$T0}); Core.Inference.return_type(op, Tuple{T1,T1})))

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

    return quote
        @_inline_meta
        @inbounds return similar_type(a, $T, Size($Snew))(tuple($(exprs...)))
    end
end

@generated function _mapreducedim(f, op, ::Size{S}, a::StaticArray, ::Type{Val{D}}, v0::T) where {S,D,T}
    N = length(S)
    Snew = ([n==D ? 1 : S[n] for n = 1:N]...)

    exprs = Array{Expr}(Snew)
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

############
## reduce ##
############

@inline reduce(op, a::StaticArray) = mapreduce(identity, op, a)
@inline reduce(op, v0, a::StaticArray) = mapreduce(identity, op, v0, a)

###############
## reducedim ##
###############

@inline reducedim(op, a::StaticArray, ::Type{Val{D}}) where {D} = mapreducedim(identity, op, a, Val{D})
@inline reducedim(op, a::StaticArray, ::Type{Val{D}}, v0) where {D} = mapreducedim(identity, op, a, Val{D}, v0)

#######################
## related functions ##
#######################

# These are all similar in Base but not @inline'd
#
# Implementation notes:
#
# 1. When providing an initial value v0, note that its location is different in reduce and
# reducedim: v0 comes earlier than collection in reduce, whereas it is the last argument in
# reducedim.  The same difference exists between mapreduce and mapreducedim.
#
# 2. mapreduce and mapreducedim usually do not take initial value v0, because we don't
# always know the return type of an arbitrary mapping function f.  (We usually want to use
# some initial value such as one(T) or zero(T) as v0, where T is the return type of f, but
# if users provide type-unstable f, its return type cannot be known.)  Therefore, mapped
# versions of the functions implemented below usually require the collection to have at
# least two entries.
#
# 3. Exceptions are the ones that require Boolean mapping functions.  For example, f in
# all and any must return Bool, so we know the appropriate v0 is true and false,
# respectively.  Therefore, all(f, ...) and any(f, ...) are implemented by mapreduce(f, ...)
# with an initial value v0 = true and false.
@inline iszero(a::StaticArray{<:Any,T}) where {T} = reduce((x,y) -> x && (y==zero(T)), true, a)

@inline sum(a::StaticArray{<:Any,T}) where {T} = reduce(+, zero(T), a)
@inline sum(f::Function, a::StaticArray) = mapreduce(f, +, a)
@inline sum(a::StaticArray{<:Any,T}, ::Type{Val{D}}) where {T,D} = reducedim(+, a, Val{D}, zero(T))
@inline sum(f::Function, a::StaticArray, ::Type{Val{D}}) where D = mapreducedim(f, +, a, Val{D})

@inline prod(a::StaticArray{<:Any,T}) where {T} = reduce(*, one(T), a)
@inline prod(f::Function, a::StaticArray{<:Any,T}) where {T} = mapreduce(f, *, a)
@inline prod(a::StaticArray{<:Any,T}, ::Type{Val{D}}) where {T,D} = reducedim(*, a, Val{D}, one(T))
@inline prod(f::Function, a::StaticArray{<:Any,T}, ::Type{Val{D}}) where {T,D} = mapreducedim(f, *, a, Val{D})

@inline count(a::StaticArray{<:Any,Bool}) = reduce(+, 0, a)
@inline count(f::Function, a::StaticArray) = mapreduce(x->f(x)::Bool, +, 0, a)
@inline count(a::StaticArray{<:Any,Bool}, ::Type{Val{D}}) where {D} = reducedim(+, a, Val{D}, 0)
@inline count(f::Function, a::StaticArray, ::Type{Val{D}}) where {D} = mapreducedim(x->f(x)::Bool, +, a, Val{D}, 0)

@inline all(a::StaticArray{<:Any,Bool}) = reduce(&, true, a)  # non-branching versions
@inline all(f::Function, a::StaticArray) = mapreduce(x->f(x)::Bool, &, true, a)
@inline all(a::StaticArray{<:Any,Bool}, ::Type{Val{D}}) where {D} = reducedim(&, a, Val{D}, true)
@inline all(f::Function, a::StaticArray, ::Type{Val{D}}) where {D} = mapreducedim(x->f(x)::Bool, &, a, Val{D}, true)

@inline any(a::StaticArray{<:Any,Bool}) = reduce(|, false, a) # (benchmarking needed)
@inline any(f::Function, a::StaticArray) = mapreduce(x->f(x)::Bool, |, false, a) # (benchmarking needed)
@inline any(a::StaticArray{<:Any,Bool}, ::Type{Val{D}}) where {D} = reducedim(|, a, Val{D}, false)
@inline any(f::Function, a::StaticArray, ::Type{Val{D}}) where {D} = mapreducedim(x->f(x)::Bool, |, a, Val{D}, false)

@inline mean(a::StaticArray) = sum(a) / length(a)
@inline mean(f::Function, a::StaticArray) = sum(f, a) / length(a)
@inline mean(a::StaticArray, ::Type{Val{D}}) where {D} = sum(a, Val{D}) / size(a, D)
@inline mean(f::Function, a::StaticArray, ::Type{Val{D}}) where {D} = sum(f, a, Val{D}) / size(a, D)

@inline minimum(a::StaticArray) = reduce(min, a) # base has mapreduce(idenity, scalarmin, a)
@inline minimum(f::Function, a::StaticArray) = mapreduce(f, min, a)
@inline minimum(a::StaticArray, ::Type{Val{D}}) where {D} = reducedim(min, a, Val{D})
@inline minimum(f::Function, a::StaticArray, ::Type{Val{D}}) where {D} = mapreducedim(f, min, a, Val{D})

@inline maximum(a::StaticArray) = reduce(max, a) # base has mapreduce(idenity, scalarmax, a)
@inline maximum(f::Function, a::StaticArray) = mapreduce(f, max, a)
@inline maximum(a::StaticArray, ::Type{Val{D}}) where {D} = reducedim(max, a, Val{D})
@inline maximum(f::Function, a::StaticArray, ::Type{Val{D}}) where {D} = mapreducedim(f, max, a, Val{D})

# Diff is slightly different
@inline diff(a::StaticArray) = diff(a, Val{1})
@inline diff(a::StaticArray, ::Type{Val{D}}) where {D} = _diff(Size(a), a, Val{D})

@generated function _diff(::Size{S}, a::StaticArray, ::Type{Val{D}}) where {S,D}
    N = length(S)
    Snew = ([n==D ? S[n]-1 : S[n] for n = 1:N]...)

    exprs = Array{Expr}(Snew)
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
