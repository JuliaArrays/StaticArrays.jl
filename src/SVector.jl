"""
    SVector{S, T}(x::NTuple{S, T})
    SVector{S, T}(x1, x2, x3, ...)

Construct a statically-sized vector `SVector`. Since this type is immutable,
the data must be provided upon construction and cannot be mutated later.
Constructors may drop the `T` and `S` parameters if they are inferrable from the
input (e.g. `SVector(1,2,3)` constructs an `SVector{3, Int}`).

    SVector{S}(vec::Vector)

Construct a statically-sized vector of length `S` using the data from `vec`.
The parameter `S` is mandatory since the length of `vec` is unknown to the
compiler (the element type may optionally also be specified).
"""
const SVector{S, T} = SArray{Tuple{S}, T, 1, S}

# Some more advanced constructor-like functions
@inline zeros(::Type{SVector{N}}) where {N} = zeros(SVector{N,Float64})
@inline ones(::Type{SVector{N}}) where {N} = ones(SVector{N,Float64})

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
        return :($SV{$(length(ex.args))}(tuple($(ex.args...))))
    elseif head === :ref
        return :($SV{$(length(ex.args)-1),$(ex.args[1])}(tuple($(ex.args[2:end]...))))
    elseif head === :typed_vcat || head === :typed_hcat || head === :typed_ncat # typed, cat
        args = parse_cat_ast(ex)
        len = check_vector_length(size(args))
        return :($SV{$len,$(ex.args[1])}(tuple($(args...))))
    elseif head === :vcat || head === :hcat || head === :ncat # untyped, cat
        args = parse_cat_ast(ex)
        len = check_vector_length(size(args))
        return :($SV{$len}(tuple($(args...))))
    elseif head === :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i) for i = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @$SV")
        end
        rng = Core.eval(mod, ex.args[2].args[2])
        exprs = (:(f($j)) for j in rng)
        return quote
            let f($(ex.args[2].args[1])) = $(ex.args[1])
                $SV{$(length(rng))}(tuple($(exprs...)))
            end
        end
    elseif head === :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i) for i = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 2
            error("Use a one-dimensional comprehension for @$SV")
        end
        rng = Core.eval(mod, ex.args[2].args[2])
        exprs = (:(f($j)) for j in rng)
        return quote
            let f($(ex.args[2].args[1])) = $(ex.args[1])
                $SV{$(length(rng)),$T}(tuple($(exprs...)))
            end
        end
    elseif head === :call
        f = ex.args[1]
        if f === :zeros || f === :ones || f === :rand || f === :randn || f === :randexp
            if length(ex.args) == 2
                return :($f($SV{$(ex.args[2])}))
            elseif length(ex.args) == 3
                return :($f($SV{$(ex.args[3:-1:2]...)}))
            else
                error("@$SV expected a 1-dimensional array expression")
            end
        elseif ex.args[1] === :fill
            if length(ex.args) == 3
                return :($f($(ex.args[2]), $SV{$(ex.args[3])}))
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
    esc(static_vector_gen(SVector, ex, __module__))
end

