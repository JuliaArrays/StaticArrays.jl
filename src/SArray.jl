"""
    SArray{Size, T, L}(x::NTuple{L, T})
    SArray{Size, T, L}(x1, x2, x3, ...)

Construct a statically-sized array `SArray`. Since this type is immutable,
the data must be provided upon construction and cannot be mutated later. The
`Size` parameter is a Tuple specifying the dimensions of the array. The
`L` parameter is the `length` of the array and is always equal to `prod(S)`.
Constructors may drop the `L` and `T` parameters if they are inferrable
from the input (e.g. `L` is always inferrable from `Size`).

    SArray{Size}(a::Array)

Construct a statically-sized array of dimensions `Size` using the data from
`a`. The `Size` parameter is mandatory since the size of `a` is unknown to the
compiler (the element type may optionally also be specified).
"""
immutable SArray{Size, T, N, L} <: StaticArray{T, N}
    data::NTuple{L,T}

    function SArray(x::NTuple{L,T})
        check_sarray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(x)
    end

    function SArray(x::NTuple{L})
        check_sarray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(convert_ntuple(T, x))
    end
end

@generated function check_sarray_parameters{Size,T,N,L}(::Type{Val{Size}}, ::Type{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if !(isa(Size, Tuple{Vararg{Int}}))
        error("SArray parameter Size must be a tuple of Ints (e.g. `SArray{(3,3)}`)")
    end

    if L != prod(Size) || L < 0 || minimum(Size) < 0 || length(Size) != N
        error("Size mismatch in SArray parameters. Got size $Size, dimension $N and length $L.")
    end

    return nothing
end

@generated function (::Type{SArray{Size,T,N}}){Size,T,N}(x::Tuple)
    return quote
        $(Expr(:meta, :inline))
        SArray{Size,T,N,$(prod(Size))}(x)
    end
end

@generated function (::Type{SArray{Size,T}}){Size,T}(x::Tuple)
    return quote
        $(Expr(:meta, :inline))
        SArray{Size,T,$(length(Size)),$(prod(Size))}(x)
    end
end

@generated function (::Type{SArray{Size}}){Size, T <: Tuple}(x::T)
    return quote
        $(Expr(:meta, :inline))
        SArray{Size,$(promote_tuple_eltype(T)),$(length(Size)),$(prod(Size))}(x)
    end
end

@inline SArray(a::StaticArray) = SArray{size(typeof(a))}(Tuple(a))


# Some more advanced constructor-like functions
@inline eye{Size}(::Type{SArray{Size}}) = eye(SArray{Size,Float64})
@inline zeros{Size}(::Type{SArray{Size}}) = zeros(SArray{Size,Float64})
@inline ones{Size}(::Type{SArray{Size}}) = ones(SArray{Size,Float64})


####################
## SArray methods ##
####################

@pure size{Size}(::Type{SArray{Size}}) = Size
@pure size{Size,T}(::Type{SArray{Size,T}}) = Size
@pure size{Size,T,N}(::Type{SArray{Size,T,N}}) = Size
@pure size{Size,T,N,L}(::Type{SArray{Size,T,N,L}}) = Size

function getindex(v::SArray, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@inline Tuple(v::SArray) = v.data

@inline function Base.unsafe_convert{Size,T}(::Type{Ptr{T}}, a::SArray{Size,T})
    Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(a))
end

macro SArray(ex)
    if !isa(ex, Expr)
        error("Bad input for @SArray")
    end

    if ex.head == :vect  # vector
        return esc(Expr(:call, SArray{(length(ex.args),)}, Expr(:tuple, ex.args...)))
    elseif ex.head == :ref # typed, vector
        return esc(Expr(:call, Expr(:curly, :SArray, ((length(ex.args)-1),), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, SArray{(s1, s2)}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :SArray, (s1, s2), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
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
            return esc(Expr(:call, SArray{(s1, s2)}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, SArray{(length(ex.args), 1)}, Expr(:tuple, ex.args...)))
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
            return esc(Expr(:call, Expr(:curly, :SArray, (s1, s2), ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :SArray, (length(ex.args)-1, 1), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        n_rng = length(ex.args) - 1
        rng_args = [ex.args[i+1].args[1] for i = 1:n_rng]
        rngs = [eval(current_module(), ex.args[i+1].args[2]) for i = 1:n_rng]
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
            $(Expr(:meta, :inline))
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SArray, (rng_lengths...)), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        n_rng = length(ex.args) - 1
        rng_args = [ex.args[i+1].args[1] for i = 1:n_rng]
        rngs = [eval(current_module(), ex.args[i+1].args[2]) for i = 1:n_rng]
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
            $(Expr(:meta, :inline))
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SArray, (rng_lengths...), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn
            if length(ex.args) == 1
                error("@SArray got bad expression: $(ex.args[1])()")
            else
                return quote
                    $(Expr(:meta, :inline))
                    if isa($(esc(ex.args[2])), DataType)
                        $(ex.args[1])($(esc(Expr(:curly, SArray, Expr(:tuple, ex.args[3:end]...), ex.args[2]))))
                    else
                        $(ex.args[1])($(esc(Expr(:curly, SArray, Expr(:tuple, ex.args[2:end]...)))))
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
                    $(Expr(:meta, :inline))
                    $(esc(ex.args[1]))($(esc(ex.args[2])), SArray{$(esc(Expr(:tuple, ex.args[3:end]...)))})
                end
            end
        elseif ex.args[1] == :eye
            if length(ex.args) == 2
                return quote
                    $(Expr(:meta, :inline))
                    eye(SArray{($(esc(ex.args[2])), $(esc(ex.args[2])))})
                end
            elseif length(ex.args) == 3
                # We need a branch, depending if the first argument is a type or a size.
                return quote
                    $(Expr(:meta, :inline))
                    if isa($(esc(ex.args[2])), DataType)
                        eye(SArray{($(esc(ex.args[3])), $(esc(ex.args[3]))), $(esc(ex.args[2]))})
                    else
                        eye(SArray{($(esc(ex.args[2])), $(esc(ex.args[3])))})
                    end
                end
            elseif length(ex.args) == 4
                return quote
                    $(Expr(:meta, :inline))
                    eye(SArray{($(esc(ex.args[3])), $(esc(ex.args[4]))), $(esc(ex.args[2]))})
                end
            else
                error("Bad eye() expression for @SArray")
            end
        else
            error("@SArray only supports the zeros(), ones(), rand(), randn() and eye() functions.")
        end
    else
        error("Bad input for @SArray")
    end
end
