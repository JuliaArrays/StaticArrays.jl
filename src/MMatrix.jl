"""
    MMatrix{S1, S2, T, L}(undef)
    MMatrix{S1, S2, T, L}(x::NTuple{L, T})
    MMatrix{S1, S2, T, L}(x1, x2, x3, ...)

Construct a statically-sized, mutable matrix `MMatrix`. The data may optionally
be provided upon construction and can be mutated later. The `L` parameter is the
`length` of the array and is always equal to `S1 * S2`. Constructors may drop
the `L`, `T` and even `S2` parameters if they are inferrable from the input
(e.g. `L` is always inferrable from `S1` and `S2`).

    MMatrix{S1, S2}(mat::Matrix)

Construct a statically-sized, mutable matrix of dimensions `S1 × S2` using the data from
`mat`. The parameters `S1` and `S2` are mandatory since the size of `mat` is
unknown to the compiler (the element type may optionally also be specified).
"""
const MMatrix{S1, S2, T, L} = MArray{Tuple{S1, S2}, T, 2, L}

@generated function (::Type{MMatrix{S1}})(x::NTuple{L}) where {S1,L}
    S2 = div(L, S1)
    if S1*S2 != L
        throw(DimensionMismatch("Incorrect matrix sizes. $S1 does not divide $L elements"))
    end
    return quote
        $(Expr(:meta, :inline))
        T = eltype(typeof(x))
        MMatrix{S1, $S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2}})(x::NTuple{L}) where {S1,S2,L}
    return quote
        $(Expr(:meta, :inline))
        T = eltype(typeof(x))
        MMatrix{S1, S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2,T}})(x::NTuple{L}) where {S1,S2,T,L}
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, L}(x)
    end
end

@generated function (::Type{MMatrix{S1,S2,T}})(::UndefInitializer) where {S1,S2,T}
    return quote
        $(Expr(:meta, :inline))
        MMatrix{S1, S2, T, $(S1*S2)}(undef)
    end
end

@inline convert(::Type{MMatrix{S1,S2}}, a::StaticArray{<:Tuple, T}) where {S1,S2,T} = MMatrix{S1,S2,T}(Tuple(a))
@inline MMatrix(a::StaticMatrix{N,M,T}) where {N,M,T} = MMatrix{N,M,T}(Tuple(a))

# Some more advanced constructor-like functions
@inline one(::Type{MMatrix{N}}) where {N} = one(MMatrix{N,N})


#####################
## MMatrix methods ##
#####################

macro MMatrix(ex)
    if !isa(ex, Expr)
        error("Bad input for @MMatrix")
    end
    if ex.head == :vect && length(ex.args) == 1 # 1 x 1
        return esc(Expr(:call, MMatrix{1, 1}, Expr(:tuple, ex.args[1])))
    elseif ex.head == :ref && length(ex.args) == 2 # typed, 1 x 1
        return esc(Expr(:call, Expr(:curly, :MMatrix, 1, 1, ex.args[1]), Expr(:tuple, ex.args[2])))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, MMatrix{s1, s2}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :MMatrix, s1, s2, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :vcat
        if isa(ex.args[1], Expr) && ex.args[1].head == :row # n x m
            # Validate
            s1 = length(ex.args)
            s2s = map(i -> ((isa(ex.args[i], Expr) && ex.args[i].head == :row) ? length(ex.args[i].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                throw(ArgumentError("Rows must be of matching lengths"))
            end

            exprs = [ex.args[i].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, MMatrix{s1, s2}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, MMatrix{length(ex.args), 1}, Expr(:tuple, ex.args...)))
        end
    elseif ex.head == :typed_vcat
        if isa(ex.args[2], Expr) && ex.args[2].head == :row # typed, n x m
            # Validate
            s1 = length(ex.args) - 1
            s2s = map(i -> ((isa(ex.args[i+1], Expr) && ex.args[i+1].head == :row) ? length(ex.args[i+1].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                throw(ArgumentError("Rows must be of matching lengths"))
            end

            exprs = [ex.args[i+1].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, Expr(:curly, :MMatrix,s1, s2, ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :MMatrix, length(ex.args)-1, 1, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @MMatrx")
        end

        rng1 = Core.eval(__module__, ex.args[2].args[2])
        rng2 = Core.eval(__module__, ex.args[3].args[2])
        f = gensym()
        f_expr = :($f = (($(ex.args[2].args[1]), $(ex.args[3].args[1])) -> $(ex.args[1])))
        exprs = [:($f($j1, $j2)) for j1 in rng1, j2 in rng2]

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MMatrix, length(rng1), length(rng2)), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @MMatrx")
        end

        rng1 = Core.eval(__module__, ex.args[2].args[2])
        rng2 = Core.eval(__module__, ex.args[3].args[2])
        f = gensym()
        f_expr = :($f = (($(ex.args[2].args[1]), $(ex.args[3].args[1])) -> $(ex.args[1])))
        exprs = [:($f($j1, $j2)) for j1 in rng1, j2 in rng2]

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MMatrix, length(rng1), length(rng2), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn || ex.args[1] == :randexp
            if length(ex.args) == 3
                return quote
                    $(ex.args[1])(MMatrix{$(esc(ex.args[2])),$(esc(ex.args[3]))})
                end
            elseif length(ex.args) == 4
                return quote
                    $(ex.args[1])(MMatrix{$(esc(ex.args[3])), $(esc(ex.args[4])), $(esc(ex.args[2]))})
                end
            else
                error("@MMatrix expected a 2-dimensional array expression")
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 4
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), MMatrix{$(esc(ex.args[3])), $(esc(ex.args[4]))})
                end
            else
                error("@MMatrix expected a 2-dimensional array expression")
            end
        else
            error("@MMatrix only supports the zeros(), ones(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Bad input for @MMatrix")
    end
end
