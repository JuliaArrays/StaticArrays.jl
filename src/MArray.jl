"""
    MArray{Size, T, L}()
    MArray{Size, T, L}(x::NTuple{L, T})
    MArray{Size, T, L}(x1, x2, x3, ...)

Construct a statically-sized, mutable array `MArray`. The data may optionally be
provided upon construction and can be mutated later. The `Size` parameter is a
Tuple specifying the dimensions of the array. The `L` parameter is the `length`
of the array and is always equal to `prod(S)`. Constructors may drop the `L` and
`T` parameters if they are inferrable from the input (e.g. `L` is always
inferrable from `Size`).

    MArray{Size}(a::Array)

Construct a statically-sized, mutable array of dimensions `Size` using the data from
`a`. The `Size` parameter is mandatory since the size of `a` is unknown to the
compiler (the element type may optionally also be specified).
"""
type MArray{Size, T, N, L} <: StaticArray{T, N}
    data::NTuple{L,T}

    function MArray(x::NTuple{L,T})
        check_marray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(x)
    end

    function MArray(x::NTuple{L,Any})
        check_marray_parameters(Val{Size}, T, Val{N}, Val{L})
        new(convert_ntuple(T, x))
    end

    function MArray()
        check_marray_parameters(Val{Size}, T, Val{N}, Val{L})
        new()
    end
end

@generated function check_marray_parameters{Size,T,N,L}(::Type{Val{Size}}, ::Type{T}, ::Type{Val{N}}, ::Type{Val{L}})
    if !(isa(Size, Tuple{Vararg{Int}}))
        error("MArray parameter Size must be a tuple of Ints (e.g. `MArray{(3,3)}`)")
    end

    if L != prod(Size) || L < 0 || minimum(Size) < 0 || length(Size) != N
        error("Size mismatch in MArray parameters. Got size $Size, dimension $N and length $L.")
    end

    return nothing
end

@generated function (::Type{MArray{Size,T,N}}){Size,T,N}(x::Tuple)
    return quote
        $(Expr(:meta, :inline))
        MArray{Size,T,N,$(prod(Size))}(x)
    end
end

@generated function (::Type{MArray{Size,T}}){Size,T}(x::Tuple)
    return quote
        $(Expr(:meta, :inline))
        MArray{Size,T,$(length(Size)),$(prod(Size))}(x)
    end
end

@generated function (::Type{MArray{Size}}){Size, T <: Tuple}(x::T)
    return quote
        $(Expr(:meta, :inline))
        MArray{Size,$(promote_tuple_eltype(T)),$(length(Size)),$(prod(Size))}(x)
    end
end

@generated function (::Type{MArray{Size,T,N}}){Size,T,N}()
    return quote
        $(Expr(:meta, :inline))
        MArray{Size, T, N, $(prod(Size))}()
    end
end

@generated function (::Type{MArray{Size,T}}){Size,T}()
    return quote
        $(Expr(:meta, :inline))
        MArray{Size, T, $(length(Size)), $(prod(Size))}()
    end
end

@inline MArray(a::StaticArray) = MArray{size(typeof(a))}(Tuple(a))

# Some more advanced constructor-like functions
@inline eye{Size}(::Type{MArray{Size}}) = eye(MArray{Size,Float64})
@inline zeros{Size}(::Type{MArray{Size}}) = zeros(MArray{Size,Float64})
@inline ones{Size}(::Type{MArray{Size}}) = ones(MArray{Size,Float64})


####################
## MArray methods ##
####################

@pure Size{S}(::Type{MArray{S}}) = Size(S)
@pure Size{S,T}(::Type{MArray{S,T}}) = Size(S)
@pure Size{S,T,N}(::Type{MArray{S,T,N}}) = Size(S)
@pure Size{S,T,N,L}(::Type{MArray{S,T,N,L}}) = Size(S)

function getindex(v::MArray, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@propagate_inbounds setindex!{S,T}(v::MArray{S,T}, val, i::Integer) = setindex!(v, convert(T, val), i)
@inline function setindex!{S,T}(v::MArray{S,T}, val::T, i::Integer)
    @boundscheck if i < 1 || i > length(v)
        throw(BoundsError())
    end

    if isbits(T)
        unsafe_store!(Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(v)), val, i)
    else
        # This one is unsafe (#27)
        # unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Void}}, Base.data_pointer_from_objref(v.data)), Base.data_pointer_from_objref(val), i)
        error("setindex!() with non-isbits eltype is not supported by StaticArrays. Consider using SizedArray.")
    end

    return val
end

@inline Tuple(v::MArray) = v.data

@inline function Base.unsafe_convert{Size,T}(::Type{Ptr{T}}, a::MArray{Size,T})
    Base.unsafe_convert(Ptr{T}, Base.data_pointer_from_objref(a))
end

macro MArray(ex)
    if !isa(ex, Expr)
        error("Bad input for @MArray")
    end

    if ex.head == :vect  # vector
        return esc(Expr(:call, MArray{(length(ex.args),)}, Expr(:tuple, ex.args...)))
    elseif ex.head == :ref # typed, vector
        return esc(Expr(:call, Expr(:curly, :MArray, ((length(ex.args)-1),), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, MArray{(s1, s2)}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :MArray, (s1, s2), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
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
            return esc(Expr(:call, MArray{(s1, s2)}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, MArray{(length(ex.args), 1)}, Expr(:tuple, ex.args...)))
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
            return esc(Expr(:call, Expr(:curly, :MArray, (s1, s2), ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :MArray, (length(ex.args)-1, 1), ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
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
            error("@MArray only supports up to 8-dimensional comprehensions")
        end

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MArray, (rng_lengths...)), Expr(:tuple, exprs...))))
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
            error("@MArray only supports up to 8-dimensional comprehensions")
        end

        return quote
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :MArray, (rng_lengths...), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn
            if length(ex.args) == 1
                error("@MArray got bad expression: $(ex.args[1])()")
            else
                return quote
                    if isa($(esc(ex.args[2])), DataType)
                        $(ex.args[1])($(esc(Expr(:curly, MArray, Expr(:tuple, ex.args[3:end]...), ex.args[2]))))
                    else
                        $(ex.args[1])($(esc(Expr(:curly, MArray, Expr(:tuple, ex.args[2:end]...)))))
                    end
                end
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 1
                error("@MArray got bad expression: $(ex.args[1])()")
            elseif length(ex.args) == 2
                error("@MArray got bad expression: $(ex.args[1])($(ex.args[2]))")
            else
                return quote
                    $(esc(ex.args[1]))($(esc(ex.args[2])), MArray{$(esc(Expr(:tuple, ex.args[3:end]...)))})
                end
            end
        elseif ex.args[1] == :eye
            if length(ex.args) == 2
                return quote
                    eye(MArray{($(esc(ex.args[2])), $(esc(ex.args[2])))})
                end
            elseif length(ex.args) == 3
                # We need a branch, depending if the first argument is a type or a size.
                return quote
                    if isa($(esc(ex.args[2])), DataType)
                        eye(MArray{($(esc(ex.args[3])), $(esc(ex.args[3]))), $(esc(ex.args[2]))})
                    else
                        eye(MArray{($(esc(ex.args[2])), $(esc(ex.args[3])))})
                    end
                end
            elseif length(ex.args) == 4
                return quote
                    eye(MArray{($(esc(ex.args[3])), $(esc(ex.args[4]))), $(esc(ex.args[2]))})
                end
            else
                error("Bad eye() expression for @MArray")
            end
        else
            error("@MArray only supports the zeros(), ones(), rand(), randn() and eye() functions.")
        end
    else
        error("Bad input for @MArray")
    end

end
