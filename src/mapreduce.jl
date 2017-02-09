#########
## map ##
#########

# Single input
@generated function map{T}(f, a1::StaticArray{T})
    exprs = [:(f(a1[$j])) for j = 1:length(a1)]
    return quote
        $(Expr(:meta, :inline))
        newtype = similar_type(typeof(a1), promote_op(f, T))
        @inbounds return $(Expr(:call, :newtype, Expr(:tuple, exprs...)))
    end
end

# Two inputs
@generated function map{T1,T2}(f, a1::StaticArray{T1}, a2::StaticArray{T2})
    if size(a1) != size(a2)
        error("Dimensions must match. Got sizes $(size(a1)) and $(size(a2))")
    end

    exprs = [:(f(a1[$j], a2[$j])) for j = 1:length(a1)]
    return quote
        $(Expr(:meta, :inline))
        newtype = similar_type(typeof(a1), promote_op(f, T1, T2))
        @inbounds return $(Expr(:call, :newtype, Expr(:tuple, exprs...)))
    end
end

# TODO these assume linear fast...
@generated function map{T1,T2}(f, a1::StaticArray{T1}, a2::AbstractArray{T2})
    exprs = [:(f(a1[$j], a2[$j])) for j = 1:length(a1)]
    return quote
        $(Expr(:meta, :inline))

        if size(a1) != size(a2)
            error("Dimensions must match. Got sizes $(size(a1)) and $(size(a2))")
        end

        newtype = similar_type(typeof(a1), promote_op(f, T1, T2))
        @inbounds return $(Expr(:call, :newtype, Expr(:tuple, exprs...)))
    end
end

@generated function map{T1,T2}(f, a1::AbstractArray{T1}, a2::StaticArray{T2})
    exprs = [:(f(a1[$j], a2[$j])) for j = 1:length(a2)]
    return quote
        $(Expr(:meta, :inline))

        @boundscheck if size(a1) != size(a2)
            error("Dimensions must match. Got sizes $(size(a1)) and $(size(a2))")
        end

        newtype = similar_type(typeof(a2), promote_op(f, T1, T2))
        @inbounds return $(Expr(:call, :newtype, Expr(:tuple, exprs...)))
    end
end

# TODO General case involving arbitrary many inputs?

############
## reduce ##
############
@generated function reduce(op, a::StaticArray)
    if length(a) == 1
        return :(@inbounds return a[1])
    else
        expr = :(op(a[1], a[2]))
        for j = 3:length(a)
            expr = :(op($expr, a[$j]))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $expr
        end
    end
end

@generated function reduce(op, v0, a::StaticArray)
    if length(a) == 0
        return :(v0)
    else
        expr = :(op(v0, a[1]))
        for j = 2:length(a)
            expr = :(op($expr, a[$j]))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $expr
        end
    end
end

@generated function reducedim{D}(op, a::StaticArray, ::Type{Val{D}})
    S = size(a)
    if S[D] == 1
        return :(return a)
    else
        N = ndims(a)
        Snew = ([n==D ? 1 : S[n] for n = 1:N]...)
        newtype = similar_type(a, Snew)

        exprs = Array{Expr}(Snew)
        itr = [1:n for n = Snew]
        for i = Base.product(itr...)
            ik = copy([i...])
            ik[D] = 2
            expr = :(op(a[$(i...)], a[$(ik...)]))
            for k = 3:S[D]
                ik[D] = k
                expr = :(op($expr, a[$(ik...)]))
            end

            exprs[i...] = expr
        end

        return quote
            $(Expr(:meta,:inline))
            @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
        end
    end
end

# These are all similar in Base but not @inline'd
@inline sum{T}(a::StaticArray{T}) = reduce(+, zero(T), a)
@inline prod{T}(a::StaticArray{T}) = reduce(*, one(T), a)
@inline count(a::StaticArray{Bool}) = reduce(+, 0, a)
@inline all(a::StaticArray{Bool}) = reduce(&, true, a)  # non-branching versions
@inline any(a::StaticArray{Bool}) = reduce(|, false, a) # (benchmarking needed)
@inline mean(a::StaticArray) = sum(a) / length(a)
@inline sumabs{T}(a::StaticArray{T}) = mapreduce(abs, +, zero(T), a)
@inline sumabs2{T}(a::StaticArray{T}) = mapreduce(abs2, +, zero(T), a)
@inline minimum(a::StaticArray) = reduce(min, a) # base has mapreduce(idenity, scalarmin, a)
@inline maximum(a::StaticArray) = reduce(max, a) # base has mapreduce(idenity, scalarmax, a)
@inline minimum{D}(a::StaticArray, dim::Type{Val{D}}) = reducedim(min, a, dim)
@inline maximum{D}(a::StaticArray, dim::Type{Val{D}}) = reducedim(max, a, dim)

@generated function diff{D}(a::StaticArray, ::Type{Val{D}}=Val{1})
    S = size(a)
    N = ndims(a)
    Snew = ([n==D ? S[n]-1 : S[n] for n = 1:N]...)
    newtype = similar_type(a, Snew)

    exprs = Array{Expr}(Snew)
    itr = [1:n for n = Snew]

    for i1 = Base.product(itr...)
        i2 = copy([i1...])
        i2[D] = i1[D] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end

    return quote
        $(Expr(:meta,:inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

###############
## mapreduce ##
###############

# Single array
@generated function mapreduce(f, op, a1::StaticArray)
    if length(a1) == 1
        return :(f(a1[1]))
    else
        expr = :(op(f(a1[1]), f(a1[2])))
        for j = 3:length(a1)
            expr = :(op($expr, f(a1[$j])))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $expr
        end
    end
end

@generated function mapreduce(f, op, v0, a1::StaticArray)
    if length(a1) == 0
        return :(v0)
    else
        expr = :(op(v0, f(a1[1])))
        for j = 2:length(a1)
            expr = :(op($expr, f(a1[$j])))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $expr
        end
    end
end

# Two arrays (e.g. dot has f(a,b) = a' * b, op = +)
@generated function mapreduce(f, op, a1::StaticArray, a2::StaticArray)
    if size(a1) != size(a2)
        error("Dimensions must match. Got sizes $(size(a)) and $(size(a2))")
    end

    if length(a1) == 1
        return :(f(a1[1], a2[1]))
    else
        expr = :(op(f(a1[1], a2[1]), f(a1[2], a2[2])))
        for j = 3:length(a1)
            expr = :(op($expr, f(a1[$j], a2[$j])))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $expr
        end
    end
end

@generated function mapreduce(f, op, v0, a1::StaticArray, a2::StaticArray)
    if size(a1) != size(a2)
        error("Dimensions must match. Got sizes $(size(a)) and $(size(a2))")
    end

    if length(a1) == 0
        return :(v0)
    else
        expr = :(op(v0, f(a1[1], a2[1])))
        for j = 2:length(a1)
            expr = :(op($expr, f(a1[$j], a2[$j])))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $expr
        end
    end
end

# TODO General case involving arbitrary many inputs?

###############
## broadcast ##
###############
# Single input version
@inline broadcast(f, a::StaticArray) = map(f, a)


# Two input versions
@generated function broadcast(f, a1::StaticArray, a2::StaticArray)
    if size(a1) == size(a2)
        return quote
            $(Expr(:meta, :inline))
            map(f, a1, a2)
        end
    else
        s1 = size(a1)
        s2 = size(a2)
        ndims = max(length(s1), length(s2))

        s = Vector{Int}(ndims)
        expands1 = Vector{Bool}(ndims)
        expands2 = Vector{Bool}(ndims)
        for i = 1:ndims
            if length(s1) < i
                s[i] = s2[i]
                expands1[i] = false
                expands2[i] = s2[i] > 1
            elseif length(s2) < i
                s[i] = s1[i]
                expands1[i] = s1[i] > 1
                expands2[i] = false
            else
                s[i] = max(s1[i], s1[i])
                @assert s1[i] == 1 || s1[i] == s[i]
                @assert s2[i] == 1 || s2[i] == s[i]
                expands1[i] = s1[i] > 1
                expands2[i] = s2[i] > 1
            end
        end
        s = (s...)
        L = prod(s)

        if s == s1
            newtype = :( similar_type($a1, promote_op(f, $(eltype(a1)), $(eltype(a2)))) )
        else
            newtype = :( similar_type($a1, promote_op(f, $(eltype(a1)), $(eltype(a2))), $s) )
        end

        exprs = Vector{Expr}(L)

        i = 1
        ind = ones(Int, ndims)
        while i <= L
            ind1 = [expands1[j] ? ind[j] : 1 for j = 1:length(s1)]
            ind2 = [expands2[j] ? ind[j] : 1 for j = 1:length(s2)]

            exprs[i] = Expr(:call, :f, Expr(:ref, :a1, ind1...), Expr(:ref, :a2, ind2...))

            i += 1

            ind[1] += 1
            j = 1
            while j < length(s)
                if ind[j] > s[j]
                    ind[j] = 1
                    ind[j+1] += 1
                else
                    break
                end

                j += 1
            end
        end

        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
        end
    end
end

@inline broadcast(f, a::StaticArray, n::Number) = map(x -> f(x, n), a)
@inline broadcast(f, n::Number, a::StaticArray) = map(x -> f(n, x), a)

# Other two-input versions with AbstractArray

##########
## map! ##
##########

# Single input
@generated function map!{F}(f::F, out::StaticArray, a1::StaticArray)
    exprs = [:(out[$j] = f(a1[$j])) for j = 1:length(a1)]
    return quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:block, exprs...))
    end
end

# Two inputs
@generated function map!{F}(f::F, out::StaticArray, a1::StaticArray, a2::StaticArray)
    if size(a1) != size(a2)
        error("Dimensions must match. Got sizes $(size(a)) and $(size(a2))")
    end

    if size(a1) != size(a2)
        error("Dimensions must match. Got sizes $(size(out)) and $(size(a1))")
    end

    exprs = [:(out[$j] = f(a1[$j], a2[$j])) for j = 1:length(a1)]
    return quote
        #$(Expr(:meta, :inline))
        @inbounds $(Expr(:block, exprs...))
    end
end


################
## broadcast! ##
################

@inline broadcast!{F}(f::F, out::StaticArray, a::StaticArray) = map!(f, out, a)
@inline broadcast!(f::typeof(identity), out::StaticArray, a::StaticArray) = map!(f, out, a)

# Two input versions
@generated function broadcast!{F}(f::F, out::StaticArray, a1::StaticArray, a2::StaticArray)
    if size(a1) == size(a2) && size(out) == size(a1)
        return quote
            $(Expr(:meta, :inline))
            @inbounds map!(f, out, a1, a2)
        end
    else
        s1 = size(a1)
        s2 = size(a2)
        ndims = max(length(s1), length(s2))

        s = Vector{Int}(ndims)
        expands1 = Vector{Bool}(ndims)
        expands2 = Vector{Bool}(ndims)
        for i = 1:ndims
            if length(s1) < i
                s[i] = s2[i]
                expands1[i] = false
                expands2[i] = s2[i] > 1
            elseif length(s2) < i
                s[i] = s1[i]
                expands1[i] = s1[i] > 1
                expands2[i] = false
            else
                s[i] = max(s1[i], s1[i])
                @assert s1[i] == 1 || s1[i] == s[i]
                @assert s2[i] == 1 || s2[i] == s[i]
                expands1[i] = s1[i] > 1
                expands2[i] = s2[i] > 1
            end
        end
        s = (s...)
        L = prod(s)

        if s != size(out)
            error("Dimension mismatch")
        end

        exprs = Vector{Expr}(L)

        i = 1
        ind = ones(Int, ndims)
        while i <= L
            ind1 = [expands1[j] ? ind[j] : 1 for j = 1:length(s1)]
            ind2 = [expands2[j] ? ind[j] : 1 for j = 1:length(s2)]
            index1 = sub2ind(s1, ind1...)
            index2 = sub2ind(s2, ind2...)

            exprs[i] = :(out[$i] = $(Expr(:call, :f, Expr(:ref, :a1, index1), Expr(:ref, :a2, index2))))

            i += 1

            ind[1] += 1
            j = 1
            while j < length(s)
                if ind[j] > s[j]
                    ind[j] = 1
                    ind[j+1] += 1
                else
                    break
                end

                j += 1
            end
        end

        return quote
            $(Expr(:meta, :inline))
            @inbounds $(Expr(:block, exprs...))
        end
    end
end

#############
## colwise ##
#############
@generated function colwise(f, vec::StaticVector, mat::StaticArray)
    length(vec) == size(mat, 1) || throw(DimensionMismatch())
    exprs = [:(f(vec, mat[:, $j])) for j = 1:size(mat, 2)]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, hcat, exprs...))
    end
end

@generated function colwise(f, mat::StaticArray, vec::StaticVector)
    length(vec) == size(mat, 1) || throw(DimensionMismatch())
    exprs = [:(f(mat[:, $j], vec)) for j = 1:size(mat, 2)]
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $(Expr(:call, hcat, exprs...))
    end
end
