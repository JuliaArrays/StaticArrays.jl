"""
    SMatrix{S1, S2, T, L}(x::NTuple{L, T})
    SMatrix{S1, S2, T, L}(x1, x2, x3, ...)

Construct a statically-sized matrix `SMatrix`. Since this type is immutable,
the data must be provided upon construction and cannot be mutated later. The
`L` parameter is the `length` of the array and is always equal to `S1 * S2`.
Constructors may drop the `L`, `T` and even `S2` parameters if they are inferrable
from the input (e.g. `L` is always inferrable from `S1` and `S2`).

    SMatrix{S1, S2}(mat::Matrix)

Construct a statically-sized matrix of dimensions `S1 × S2` using the data from
`mat`. The parameters `S1` and `S2` are mandatory since the size of `mat` is
unknown to the compiler (the element type may optionally also be specified).
"""
const SMatrix{S1, S2, T, L} = SArray{Tuple{S1, S2}, T, 2, L}

# Some more advanced constructor-like functions
@inline one(::Type{SMatrix{N}}) where {N} = one(SMatrix{N,N})

#####################
## SMatrix methods ##
#####################

function check_matrix_size(x::Tuple, T = :S)
    if length(x) > 2
        all(isone, x[3:end]) || error("Bad input for @$(T)Matrix, must be matrix like.")
    end
    x1 = length(x) >= 1 ? x[1] : 1
    x2 = length(x) >= 2 ? x[2] : 1
    x1, x2
end

function static_matrix_gen(::Type{SM}, @nospecialize(ex), mod::Module) where {SM}
    if !isa(ex, Expr)
        error("Bad input for @$SM")
    end
    head = ex.head
    if head === :vect && length(ex.args) == 1 # 1 x 1
        return :($SM{1,1}(tuple($(ex.args[1]))))
    elseif head === :ref && length(ex.args) == 2 # typed, 1 x 1
        return :($SM{1,1,$(ex.args[1])}(tuple($(ex.args[2]))))
    elseif head === :typed_vcat || head === :typed_hcat || head === :typed_ncat # typed, cat
        args = parse_cat_ast(ex)
        sz1, sz2 = check_matrix_size(size(args))
        return :($SM{$sz1,$sz2,$(ex.args[1])}(tuple($(args...))))
    elseif head === :vcat || head === :hcat || head === :ncat # untyped, cat
        args = parse_cat_ast(ex)
        sz1, sz2 = check_matrix_size(size(args))
        return :($SM{$sz1,$sz2}(tuple($(args...))))
    elseif head === :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || (ex.args[1]::Expr).head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @$SM")
        end
        rng1 = Core.eval(mod, ex.args[2].args[2])
        rng2 = Core.eval(mod, ex.args[3].args[2])
        exprs = (:(f($j1, $j2)) for j1 in rng1, j2 in rng2)
        return quote
            let f($(ex.args[2].args[1]), $(ex.args[3].args[1])) = $(ex.args[1])
                $SM{$(length(rng1)),$(length(rng2))}(tuple($(exprs...)))
            end
        end
    elseif head === :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || (ex.args[2]::Expr).head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @$SM")
        end
        rng1 = Core.eval(mod, ex.args[2].args[2])
        rng2 = Core.eval(mod, ex.args[3].args[2])
        exprs = (:(f($j1, $j2)) for j1 in rng1, j2 in rng2)
        return quote
            let f($(ex.args[2].args[1]), $(ex.args[3].args[1])) = $(ex.args[1])
                $SM{$(length(rng1)),$(length(rng2)),$T}(tuple($(exprs...)))
            end
        end
    elseif head === :call
        f = ex.args[1]
        if f === :zeros || f === :ones || f === :rand || f === :randn || f === :randexp
            if length(ex.args) == 3
                return :($f($SM{$(ex.args[2:3]...)}))
            elseif length(ex.args) == 4
                return :($f($SM{$(ex.args[[3,4,2]]...)}))
            else
                error("@$SM expected a 2-dimensional array expression")
            end
        elseif ex.args[1] === :fill
            if length(ex.args) == 4
                return :($f($(ex.args[2]), $SM{$(ex.args[3:4]...)}))
            else
                error("@$SM expected a 2-dimensional array expression")
            end
        else
            error("@$SM only supports the zeros(), ones(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Bad input for @$SM")
    end
end


"""
    @SMatrix [a b c d]
    @SMatrix [[a, b];[c, d]]
    @SMatrix [i+j for i in 1:2, j in 1:2]
    @SMatrix ones(2, 2)

A convenience macro to construct `SMatrix`.
See [`@SArray`](@ref) for detailed features.
"""
macro SMatrix(ex)
    esc(static_matrix_gen(SMatrix, ex, __module__))
end
