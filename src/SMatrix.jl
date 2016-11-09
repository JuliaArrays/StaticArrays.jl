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
immutable SMatrix{S1, S2, T, L} <: StaticMatrix{T}
    data::NTuple{L, T}

    function SMatrix(d::NTuple{L,T})
        check_smatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new(d)
    end

    function SMatrix(d::NTuple{L})
        check_smatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new(convert_ntuple(T, d))
    end
end

@generated function check_smatrix_params{S1,S2,L}(::Type{Val{S1}}, ::Type{Val{S2}}, T, ::Type{Val{L}})
    if !(T <: DataType) # I think the way types are handled in generated fnctions might have changed in 0.5?
        return :(error("SMatrix: Parameter T must be a DataType. Got $T"))
    end

    if !isa(S1, Int) || !isa(S2, Int) || !isa(L, Int) || S1 < 0 || S2 < 0 || L < 0
        return :(error("SMatrix: Sizes must be positive integers. Got $S1 × $S2 ($L elements)"))
    end

    if S1*S2 == L
        return nothing
    else
        str = "Size mismatch in SMatrix. S1 = $S1, S2 = $S2, but recieved $L elements"
        return :(error($str))
    end
end

@generated function (::Type{SMatrix{S1}}){S1,L}(x::NTuple{L})
    S2 = div(L, S1)
    if S1*S2 != L
        error("Incorrect matrix sizes. $S1 does not divide $L elements")
    end
    T = promote_tuple_eltype(x)

    return quote
        $(Expr(:meta, :inline))
        SMatrix{S1, $S2, $T, L}(x)
    end
end

@generated function (::Type{SMatrix{S1,S2}}){S1,S2,L}(x::NTuple{L})
    T = promote_tuple_eltype(x)

    return quote
        $(Expr(:meta, :inline))
        SMatrix{S1, S2, $T, L}(x)
    end
end

@generated function (::Type{SMatrix{S1,S2,T}}){S1,S2,T,L}(x::NTuple{L})
    return quote
        $(Expr(:meta, :inline))
        SMatrix{S1, S2, T, L}(x)
    end
end

@inline convert{S1,S2,T}(::Type{SMatrix{S1,S2}}, a::StaticArray{T}) = SMatrix{S1,S2,T}(Tuple(a))
@inline SMatrix(a::StaticMatrix) = SMatrix{size(typeof(a),1),size(typeof(a),2)}(Tuple(a))

#=
@inline (::Type{SMatrix{S1}}){S1}(x1) = SMatrix{S1}((x1,))
@inline (::Type{SMatrix{S1}}){S1}(x1,x2) = SMatrix{S1}((x1,x2))
@inline (::Type{SMatrix{S1}}){S1}(x1,x2,x3) = SMatrix{S1}((x1,x2,x3))
@inline (::Type{SMatrix{S1}}){S1}(x1,x2,x3,x4) = SMatrix{S1}((x1,x2,x3,x4))
@inline (::Type{SMatrix{S1}}){S1}(x...) = SMatrix{S1}(x)

@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1) = SMatrix{S1,S2}((x1,))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1,x2) = SMatrix{S1,S2}((x1,x2))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1,x2,x3) = SMatrix{S1,S2}((x1,x2,x3))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1,x2,x3,x4) = SMatrix{S1,S2}((x1,x2,x3,x4))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x...) = SMatrix{S1,S2}(x)

@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1) = SMatrix{S1,S2,T}((x1,))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1,x2) = SMatrix{S1,S2,T}((x1,x2))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1,x2,x3) = SMatrix{S1,S2,T}((x1,x2,x3))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1,x2,x3,x4) = SMatrix{S1,S2,T}((x1,x2,x3,x4))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x...) = SMatrix{S1,S2,T}(x)
=#

# Some more advanced constructor-like functions
@inline one{N}(::Type{SMatrix{N}}) = one(SMatrix{N,N})
@inline eye{N}(::Type{SMatrix{N}}) = eye(SMatrix{N,N})
@inline eye{N,M}(::Type{SMatrix{N,M}}) = eye(SMatrix{N,M,Float64})
@inline zeros{N,M}(::Type{SMatrix{N,M}}) = zeros(SMatrix{N,M,Float64})
@inline ones{N,M}(::Type{SMatrix{N,M}}) = ones(SMatrix{N,M,Float64})

#####################
## SMatrix methods ##
#####################

@pure size{S1,S2}(::Type{SMatrix{S1,S2}}) = (S1, S2)
@pure size{S1,S2,T}(::Type{SMatrix{S1,S2,T}}) = (S1, S2)
@pure size{S1,S2,T,L}(::Type{SMatrix{S1,S2,T,L}}) = (S1, S2)

function getindex(v::SMatrix, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@inline Tuple(v::SMatrix) = v.data

# See #53
Base.cconvert{T}(::Type{Ptr{T}}, m::SMatrix) = Ref(m)
Base.unsafe_convert{N,M,T,L}(::Type{Ptr{T}}, m::Ref{SMatrix{N,M,T,L}}) =
    Ptr{T}(Base.unsafe_convert(Ptr{SMatrix{N,M,T,L}}, m))


macro SMatrix(ex)
    if !isa(ex, Expr)
        error("Bad input for @SMatrix")
    end

    if ex.head == :vect && length(ex.args) == 1 # 1 x 1
        return esc(Expr(:call, SMatrix{1, 1}, Expr(:tuple, ex.args[1])))
    elseif ex.head == :ref && length(ex.args) == 2 # typed, 1 x 1
        return esc(Expr(:call, Expr(:curly, :SMatrix, 1, 1, ex.args[1]), Expr(:tuple, ex.args[2])))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return esc(Expr(:call, SMatrix{s1, s2}, Expr(:tuple, ex.args...)))
    elseif ex.head == :typed_hcat # typed, 1 x n
        s1 = 1
        s2 = length(ex.args) - 1
        return esc(Expr(:call, Expr(:curly, :SMatrix, s1, s2, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
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
            return esc(Expr(:call, SMatrix{s1, s2}, Expr(:tuple, exprs...)))
        else # n x 1
            return esc(Expr(:call, SMatrix{length(ex.args), 1}, Expr(:tuple, ex.args...)))
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
            return esc(Expr(:call, Expr(:curly, :SMatrix,s1, s2, ex.args[1]), Expr(:tuple, exprs...)))
        else # typed, n x 1
            return esc(Expr(:call, Expr(:curly, :SMatrix, length(ex.args)-1, 1, ex.args[1]), Expr(:tuple, ex.args[2:end]...)))
        end
    elseif isa(ex, Expr) && ex.head == :comprehension
        if length(ex.args) != 1 || !isa(ex.args[1], Expr) || ex.args[1].head != :generator
            error("Expected generator in comprehension, e.g. [f(i,j) for i = 1:3, j = 1:3]")
        end
        ex = ex.args[1]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @SMatrx")
        end

        rng1 = eval(current_module(), ex.args[2].args[2])
        rng2 = eval(current_module(), ex.args[3].args[2])
        f = gensym()
        f_expr = :($f = (($(ex.args[2].args[1]), $(ex.args[3].args[1])) -> $(ex.args[1])))
        exprs = [:($f($j1, $j2)) for j1 in rng1, j2 in rng2]

        return quote
            $(Expr(:meta, :inline))
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SMatrix, length(rng1), length(rng2)), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :typed_comprehension
        if length(ex.args) != 2 || !isa(ex.args[2], Expr) || ex.args[2].head != :generator
            error("Expected generator in typed comprehension, e.g. Float64[f(i,j) for i = 1:3, j = 1:3]")
        end
        T = ex.args[1]
        ex = ex.args[2]
        if length(ex.args) != 3
            error("Use a 2-dimensional comprehension for @SMatrx")
        end

        rng1 = eval(current_module(), ex.args[2].args[2])
        rng2 = eval(current_module(), ex.args[3].args[2])
        f = gensym()
        f_expr = :($f = (($(ex.args[2].args[1]), $(ex.args[3].args[1])) -> $(ex.args[1])))
        exprs = [:($f($j1, $j2)) for j1 in rng1, j2 in rng2]

        return quote
            $(Expr(:meta, :inline))
            $(esc(f_expr))
            $(esc(Expr(:call, Expr(:curly, :SMatrix, length(rng1), length(rng2), T), Expr(:tuple, exprs...))))
        end
    elseif isa(ex, Expr) && ex.head == :call
        if ex.args[1] == :zeros || ex.args[1] == :ones || ex.args[1] == :rand || ex.args[1] == :randn
            if length(ex.args) == 3
                return quote
                    $(Expr(:meta, :inline))
                    $(ex.args[1])(SMatrix{$(esc(ex.args[2])),$(esc(ex.args[3]))})
                end
            elseif length(ex.args) == 4
                return quote
                    $(Expr(:meta, :inline))
                    $(ex.args[1])(SMatrix{$(esc(ex.args[3])), $(esc(ex.args[4])), $(esc(ex.args[2]))})
                end
            else
                error("@SMatrix expected a 2-dimensional array expression")
            end
        elseif ex.args[1] == :fill
            if length(ex.args) == 4
                return quote
                    $(Expr(:meta, :inline))
                    $(esc(ex.args[1]))($(esc(ex.args[2])), SMatrix{$(esc(ex.args[3])), $(esc(ex.args[4]))})
                end
            else
                error("@SMatrix expected a 2-dimensional array expression")
            end
        elseif ex.args[1] == :eye
            if length(ex.args) == 2
                return quote
                    $(Expr(:meta, :inline))
                    eye(SMatrix{$(esc(ex.args[2]))})
                end
            elseif length(ex.args) == 3
                # We need a branch, depending if the first argument is a type or a size.
                return quote
                    $(Expr(:meta, :inline))
                    if isa($(esc(ex.args[2])), DataType)
                        eye(SMatrix{$(esc(ex.args[3])), $(esc(ex.args[3])), $(esc(ex.args[2]))})
                    else
                        eye(SMatrix{$(esc(ex.args[2])), $(esc(ex.args[3]))})
                    end
                end
            elseif length(ex.args) == 4
                return quote
                    $(Expr(:meta, :inline))
                    eye(SMatrix{$(esc(ex.args[3])), $(esc(ex.args[4])), $(esc(ex.args[2]))})
                end
            else
                error("Bad eye() expression for @SMatrix")
            end
        else
            error("@SMatrix only supports the zeros(), ones(), fill(), rand(), randn() and eye() functions.")
        end
    else
        error("Bad input for @SMatrix")
    end
end
