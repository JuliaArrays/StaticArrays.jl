
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

# @SMatrix rand(...)
@inline _rand_with_Val(::Type{SM}, rng::AbstractRNG, ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = rand(rng, SM{n1, n2})
@inline _rand_with_Val(::Type{SM}, T::DataType,      ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = _rand(Random.GLOBAL_RNG, T, Size(n1, n2), SM{n1, n2, T})
@inline _rand_with_Val(::Type{SM}, sampler,          ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = _rand(Random.GLOBAL_RNG, sampler, Size(n1, n2), SM{n1, n2, Random.gentype(sampler)})
@inline _rand_with_Val(::Type{SM}, rng::AbstractRNG, T::DataType, ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = rand(rng, SM{n1, n2, T})
@inline _rand_with_Val(::Type{SM}, rng::AbstractRNG, sampler,     ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = _rand(rng, sampler, Size(n1, n2), SM{n1, n2, Random.gentype(sampler)})
# @SMatrix randn(...)
@inline _randn_with_Val(::Type{SM}, rng::AbstractRNG, ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = randn(rng, SM{n1, n2})
@inline _randn_with_Val(::Type{SM}, T::DataType,      ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = _randn(Random.GLOBAL_RNG, Size(n1, n2), SM{n1, n2, T})
@inline _randn_with_Val(::Type{SM}, rng::AbstractRNG, T::DataType, ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = randn(rng, SM{n1, n2, T})
# @SMatrix randexp(...)
@inline _randexp_with_Val(::Type{SM}, rng::AbstractRNG, ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = randexp(rng, SM{n1, n2})
@inline _randexp_with_Val(::Type{SM}, T::DataType,      ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = _randexp(Random.GLOBAL_RNG, Size(n1, n2), SM{n1, n2, T})
@inline _randexp_with_Val(::Type{SM}, rng::AbstractRNG, T::DataType, ::Val{n1}, ::Val{n2}) where {SM, n1, n2} = randexp(rng, SM{n1, n2, T})

function static_matrix_gen(::Type{SM}, @nospecialize(ex), mod::Module) where {SM}
    if !isa(ex, Expr)
        error("Bad input for @$SM")
    end
    head = ex.head
    if head === :vect && length(ex.args) == 1 # 1 x 1
        return :($SM{1,1}($tuple($(esc(ex.args[1])))))
    elseif head === :ref && length(ex.args) == 2 # typed, 1 x 1
        return :($SM{1,1,$(esc(ex.args[1]))}($tuple($(esc(ex.args[2])))))
    elseif head === :typed_vcat || head === :typed_hcat || head === :typed_ncat # typed, cat
        args = parse_cat_ast(ex)
        sz1, sz2 = check_matrix_size(size(args))
        return :($SM{$sz1,$sz2,$(esc(ex.args[1]))}($tuple($(escall(args)...))))
    elseif head === :vcat || head === :hcat || head === :ncat # untyped, cat
        args = parse_cat_ast(ex)
        sz1, sz2 = check_matrix_size(size(args))
        return :($SM{$sz1,$sz2}($tuple($(escall(args)...))))
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
            let
                f($(esc(ex.args[2].args[1])), $(esc(ex.args[3].args[1]))) = $(esc(ex.args[1]))
                $SM{$(length(rng1)),$(length(rng2))}($tuple($(exprs...)))
            end
        end
    elseif head === :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || (ex.args[2]::Expr).head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = esc(ex.args[1])
        ex = ex.args[2]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @$SM")
        end
        rng1 = Core.eval(mod, ex.args[2].args[2])
        rng2 = Core.eval(mod, ex.args[3].args[2])
        exprs = (:(f($j1, $j2)) for j1 in rng1, j2 in rng2)
        return quote
            let 
                f($(esc(ex.args[2].args[1])), $(esc(ex.args[3].args[1]))) = $(esc(ex.args[1]))
                $SM{$(length(rng1)),$(length(rng2)),$T}($tuple($(exprs...)))
            end
        end
    elseif head === :call
        f = ex.args[1]
        fargs = ex.args[2:end]
        if f === :zeros || f === :ones
            if length(fargs) == 2
                # for calls like `zeros(dim1, dim2)`
                return :($f($SM{$(escall(fargs)...)}))
            elseif length(fargs[2:end]) == 2
                # for calls like `zeros(type, dim1, dim2)`
                return :($f($SM{$(escall(fargs[2:end])...), $(esc(fargs[1]))}))
            else
                error("@$SM got bad expression: $(ex)")
            end
        elseif f === :fill
            # for calls like `fill(value, dim1, dim2)`
            if length(fargs[2:end]) == 2
                return :($f($(esc(fargs[1])), $SM{$(escall(fargs[2:end])...)}))
            else
                error("@$SM expected a 2-dimensional array expression")
            end
        elseif f === :rand || f === :randn || f === :randexp
            _f_with_Val = Symbol(:_, f, :_with_Val)
            if length(fargs) == 2
                # for calls like `rand(dim1, dim2)`
                # for calls like `randn(dim1, dim2)`
                # for calls like `randexp(dim1, dim2)`
                return :($f($SM{$(escall(fargs)...)}))
            elseif length(fargs) == 3
                # for calls like `rand(rng, dim1, dim2)`
                # for calls like `rand(type, dim1, dim2)`
                # for calls like `rand(sampler, dim1, dim2)`
                # for calls like `randn(rng, dim1, dim2)`
                # for calls like `randn(type, dim1, dim2)`
                # for calls like `randexp(rng, dim1, dim2)`
                # for calls like `randexp(type, dim1, dim2)`
                return :($_f_with_Val($SM, $(esc(fargs[1])), Val($(esc(fargs[2]))), Val($(esc(fargs[3])))))
            elseif length(fargs) == 4
                # for calls like `rand(rng, type, dim1, dim2)`
                # for calls like `rand(rng, sampler, dim1, dim2)`
                # for calls like `randn(rng, type, dim1, dim2)`
                # for calls like `randexp(rng, type, dim1, dim2)`
                return :($_f_with_Val($SM, $(esc(fargs[1])), $(esc(fargs[2])), Val($(esc(fargs[3]))), Val($(esc(fargs[4])))))
            else
                error("@$SM got bad expression: $(ex)")
            end
        else
            error("@$SM only supports the zeros(), ones(), fill(), rand(), randn(), and randexp() functions.")
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
    static_matrix_gen(SMatrix, ex, __module__)
end
