"""
    SArray{S, T, L}(x::NTuple{L, T})
    SArray{S, T, L}(x1, x2, x3, ...)

Construct a statically-sized array `SArray`. Since this type is immutable, the data must be
provided upon construction and cannot be mutated later. The `S` parameter is a Tuple-type
specifying the dimensions, or size, of the array - such as `Tuple{3,4,5}` for a 3×4×5-sized
array. The `L` parameter is the `length` of the array and is always equal to `prod(S)`.
Constructors may drop the `L` and `T` parameters if they are inferrable from the input
(e.g. `L` is always inferrable from `S`).

    SArray{S}(a::Array)

Construct a statically-sized array of dimensions `S` (expressed as a `Tuple{...}`) using
the data from `a`. The `S` parameter is mandatory since the size of `a` is unknown to the
compiler (the element type may optionally also be specified).
"""
struct SArray{S <: Tuple, T, N, L} <: StaticArray{S, T, N}
    data::NTuple{L,T}

    function SArray{S, T, N, L}(x::NTuple{L,T}) where {S, T, N, L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S, T, N, L}(x)
    end

    function SArray{S, T, N, L}(x::NTuple{L,Any}) where {S, T, N, L}
        check_array_parameters(S, T, Val{N}, Val{L})
        new{S, T, N, L}(convert_ntuple(T, x))
    end
end

@generated function (::Type{SArray{S, T, N}})(x::Tuple) where {S <: Tuple, T, N}
    return quote
        @_inline_meta
        SArray{S, T, N, $(tuple_prod(S))}(x)
    end
end

@generated function (::Type{SArray{S, T}})(x::Tuple) where {S <: Tuple, T}
    return quote
        @_inline_meta
        SArray{S, T, $(tuple_length(S)), $(tuple_prod(S))}(x)
    end
end

@generated function (::Type{SArray{S}})(x::T) where {S <: Tuple, T <: Tuple}
    return quote
        @_inline_meta
        SArray{S, $(promote_tuple_eltype(T)), $(tuple_length(S)), $(tuple_prod(S))}(x)
    end
end

@inline SArray(a::StaticArray) = SArray{size_tuple(Size(a))}(Tuple(a))

# Simplified show for the type
# show(io::IO, ::Type{SArray{S, T, N}}) where {S, T, N} = print(io, "SArray{$S,$T,$N}") # TODO reinstate

# Some more advanced constructor-like functions
@inline one(::Type{SArray{S}}) where {S} = one(SArray{S, Float64, tuple_length(S)})
@inline one(::Type{SArray{S, T}}) where {S, T} = one(SArray{S, T, tuple_length(S)})

# SArray(I::UniformScaling) methods to replace eye
(::Type{SA})(I::UniformScaling) where {SA<:SArray} = _eye(Size(SA), SA, I)

####################
## SArray methods ##
####################

@propagate_inbounds function getindex(v::SArray, i::Int)
    v.data[i]
end

@inline Tuple(v::SArray) = v.data

Base.dataids(::SArray) = ()

# See #53
Base.cconvert(::Type{Ptr{T}}, a::SArray) where {T} = Base.RefValue(a)
Base.unsafe_convert(::Type{Ptr{T}}, a::Base.RefValue{SArray{S,T,D,L}}) where {S,T,D,L} =
    Ptr{T}(Base.unsafe_convert(Ptr{SArray{S,T,D,L}}, a))

macro SArray(ex)
    if !isa(ex, Expr)
        error("Bad input for @SArray")
    end

    if ex.head == :vect  # vector
        return esc(Expr(:call, SArray{Tuple{length(ex.args)}}, Expr(:tuple, ex.args...)))
    elseif ex.head == :ref # typed, vector
        return esc(Expr(:call, Expr(:curly, :SArray, Tuple{length(ex.args)-1}, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, SArray{Tuple{s1, s2}}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :SArray, Tuple{s1, s2}, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :vcat
        if isa(ex.args[1], Expr) && ex.args[1].head == :row # n x m
            # Validate
            s1 = length(ex.args)
            s2s = map(i -> ((isa(ex.args[i], Expr) && ex.args[i].head == :row) ? length(ex.args[i].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, SArray{Tuple{s1, s2}}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, SArray{Tuple{length(ex.args), 1}}, Expr(:tuple, ex.args...)))
        end
    elseif ex.head == :typed_vcat
        if isa(ex.args[2], Expr) && ex.args[2].head == :row # typed, n x m
            # Validate
            s1 = length(ex.args) - 1
            s2s = map(i -> ((isa(ex.args[i+1], Expr) && ex.args[i+1].head == :row) ? length(ex.args[i+1].args) : 1), 1:s1)
            s2 = minimum(s2s)
            if maximum(s2s) != s2
                error("Rows must be of matching lengths")
            end

            exprs = [ex.args[i+1].args[j] for i = 1:s1, j = 1:s2]
            return esc(Expr(:call, Expr(:curly, :SArray, Tuple{s1, s2}, ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :SArray, Tuple{length(ex.args)-1, 1}, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        n_rng = length(ex.args) - 1
        rng_args = [ex.args[i+1].args[1] for i = 1:n_rng]
        rngs = Any[Core.eval(__module__, ex.args[i+1].args[2]) for i = 1:n_rng]
        rng_lengths = map(length, rngs)

        f = gensym()
        f_expr = :($f = ($(Expr(:tuple, rng_args...)) -> $(ex.args[1])))

        # TODO figure out a generic way of doing this...
        if n_rng == 1
            exprs = [:($f($j1)) for j1 in rngs[1]]
        elseif n_rng == 2
            exprs = [:($f($j1, $j2)) for j1 in rngs[1], j2 in rngs[2]]
        elseif n_rng == 3
            exprs = [:($f($j1, $j2, $j3)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3]]
        elseif n_rng == 4
            exprs = [:($f($j1, $j2, $j3, $j4)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4]]
        elseif n_rng == 5
            exprs = [:($f($j1, $j2, $j3, $j4, $j5)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5]]
        elseif n_rng == 6
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6]]
        elseif n_rng == 7
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7]]
        elseif n_rng == 8
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7, $j8)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7], j8 in rngs[8]]
        else
            error("@SArray only supports up to 8-dimensional comprehensions")
        end

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SArray, Tuple{rng_lengths...}), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        n_rng = length(ex.args) - 1
        rng_args = [ex.args[i+1].args[1] for i = 1:n_rng]
        rngs = [Core.eval(__module__, ex.args[i+1].args[2]) for i = 1:n_rng]
        rng_lengths = map(length, rngs)

        f = gensym()
        f_expr = :($f = ($(Expr(:tuple, rng_args...)) -> $(ex.args[1])))

        # TODO figure out a generic way of doing this...
        if n_rng == 1
            exprs = [:($f($j1)) for j1 in rngs[1]]
        elseif n_rng == 2
            exprs = [:($f($j1, $j2)) for j1 in rngs[1], j2 in rngs[2]]
        elseif n_rng == 3
            exprs = [:($f($j1, $j2, $j3)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3]]
        elseif n_rng == 4
            exprs = [:($f($j1, $j2, $j3, $j4)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4]]
        elseif n_rng == 5
            exprs = [:($f($j1, $j2, $j3, $j4, $j5)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5]]
        elseif n_rng == 6
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6]]
        elseif n_rng == 7
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7]]
        elseif n_rng == 8
            exprs = [:($f($j1, $j2, $j3, $j4, $j5, $j6, $j7, $j8)) for j1 in rngs[1], j2 in rngs[2], j3 in rngs[3], j4 in rngs[4], j5 in rngs[5], j6 in rngs[6], j7 in rngs[7], j8 in rngs[8]]
        else
            error("@SArray only supports up to 8-dimensional comprehensions")
        end

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SArray, Tuple{rng_lengths...}, T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn || ex.args[1] == :randexp
            if length(ex.args) == 1
                error("@SArray got bad expression: $(ex.args[1])()")
            else
                return quote
                    if isa($(esc(ex.args[2])), DataType)
                        $(ex.args[1])($(esc(Expr(:curly, SArray, Expr(:curly, Tuple, ex.args[3:end]...), ex.args[2]))))
                    else
                        $(ex.args[1])($(esc(Expr(:curly, SArray, Expr(:curly, Tuple, ex.args[2:end]...)))))
                    end
                end
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 1
                error("@SArray got bad expression: $(ex.args[1])()")
            elseif length(ex.args) == 2
                error("@SArray got bad expression: $(ex.args[1])($(ex.args[2]))")
            else
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), SArray{$(esc(Expr(:curly, Tuple, ex.args[3:end]...)))})
                end
            end
        else
            error("@SArray only supports the zeros(), ones(), rand(), randn(), and randexp() functions.")
        end
    else
        error("Bad input for @SArray")
    end
end

function promote_rule(::Type{<:SArray{S,T,N,L}}, ::Type{<:SArray{S,U,N,L}}) where {S,T,U,N,L}
    SArray{S,promote_type(T,U),N,L}
end
