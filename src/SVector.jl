#####################
## SVector methods ##
#####################

# Converting a CartesianIndex to an SVector
convert(::Type{SVector}, I::CartesianIndex) = SVector(I.I)
convert(::Type{SVector{N}}, I::CartesianIndex{N}) where {N} = SVector{N}(I.I)
convert(::Type{SVector{N,T}}, I::CartesianIndex{N}) where {N,T} = SVector{N,T}(I.I)

Base.promote_rule(::Type{SVector{N,T}}, ::Type{CartesianIndex{N}}) where {N,T} = SVector{N,promote_type(T,Int)}

function check_vector_length(x::Tuple, T = :S)
    if length(x) > 1
        all(isone, Base.tail(x)) || error("Bad input for @$(T)Vector, must be vector like")
    end
    length(x) >= 1 ? x[1] : 1
end

function static_vector_gen(::Type{SV}, @nospecialize(ex), mod::Module) where {SV}
    if !isa(ex, Expr)
        error("Bad input for @$SV")
    end
    head = ex.head
    if head === :vect
        return :($SV{$(length(ex.args))}($tuple($(escall(ex.args)...))))
    elseif head === :ref
        return :($SV{$(length(ex.args)-1),$(esc(ex.args[1]))}($tuple($(escall(ex.args[2:end])...))))
    elseif head === :typed_vcat || head === :typed_hcat || head === :typed_ncat # typed, cat
        args = parse_cat_ast(ex)
        len = check_vector_length(size(args))
        return :($SV{$len,$(esc(ex.args[1]))}($tuple($(escall(args)...))))
    elseif head === :vcat || head === :hcat || head === :ncat # untyped, cat
        args = parse_cat_ast(ex)
        len = check_vector_length(size(args))
        return :($SV{$len}($tuple($(escall(args)...))))
    elseif head === :comprehension
        if length(ex.args) != 1
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        ex = ex.args[1]
        if !isa(ex, Expr) || (ex::Expr).head != :generator
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @$SV")
        end
        rng = Core.eval(mod, ex.args[2].args[2])
        exprs = (:(f($j)) for j in rng)
        return quote
            let
                f($(esc(ex.args[2].args[1]))) = $(esc(ex.args[1]))
                $SV{$(length(rng))}($tuple($(exprs...)))
            end
        end
    elseif head === :typed_comprehension
        if length(ex.args) != 2
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        T = esc(ex.args[1])
        ex = ex.args[2]
        if !isa(ex, Expr) || (ex::Expr).head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @$SV")
        end
        rng = Core.eval(mod, ex.args[2].args[2])
        exprs = (:(f($j)) for j in rng)
        return quote
            let 
                f($(esc(ex.args[2].args[1]))) = $(esc(ex.args[1]))
                $SV{$(length(rng)),$T}($tuple($(exprs...)))
            end
        end
    elseif head === :call
        f = ex.args[1]
        if f === :zeros || f === :ones || f === :rand || f === :randn || f === :randexp
            if length(ex.args) == 2
                return :($f($SV{$(esc(ex.args[2])), Float64})) # default to Float64 like Base
            elseif length(ex.args) == 3
                if f === :rand && ex.args[3] isa Int && ex.args[3] ≥ 0
                    # for calls like rand(sampler, n) or rand(type, n)
                    return quote
                        StaticArrays._rand(
                            Random.GLOBAL_RNG,
                            $(esc(ex.args[2])),
                            Size($(esc(ex.args[3]))),
                            $SV{$(esc(ex.args[3])), Random.gentype($(esc(ex.args[2])))},
                        )
                    end
                else
                    return :($f($SV{$(escall(ex.args[3:-1:2])...)}))
                end
            elseif length(ex.args) == 4 && f === :rand && ex.args[4] isa Int && ex.args[4] ≥ 0
                # for calls like rand(rng::AbstractRNG, sampler, n) or rand(rng::AbstractRNG, type, n)
                return quote 
                    StaticArrays._rand(
                        $(esc(ex.args[2])),
                        $(esc(ex.args[3])),
                        Size($(esc(ex.args[4]))),
                        $SV{$(esc(ex.args[4])), Random.gentype($(esc(ex.args[3])))},
                    )
                end
            else
                error("@$SV expected a 1-dimensional array expression")
            end
        elseif ex.args[1] === :fill
            if length(ex.args) == 3
                return :($f($(esc(ex.args[2])), $SV{$(esc(ex.args[3]))}))
            else
                error("@$SV expected a 1-dimensional array expression")
            end
        else
            error("@$SV only supports the zeros(), ones(), rand(), randn() and randexp() functions.")
        end
    else
        error("Use @$SV [a,b,c], @$SV Type[a,b,c] or a comprehension like @$SV [f(i) for i = i_min:i_max]")
    end
end

"""
    @SVector [a, b, c, d]
    @SVector [i for i in 1:2]
    @SVector ones(2)

A convenience macro to construct `SVector`.
See [`@SArray`](@ref) for detailed features.
"""
macro SVector(ex)
    static_vector_gen(SVector, ex, __module__)
end

