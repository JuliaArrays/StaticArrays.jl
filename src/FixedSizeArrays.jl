"""
Compatibility layer for transferring from FixedSizeArrays. This provides
alternative definitions of `Vec`, `Mat`, `Point`, `FixedVectorNoTuple`, `@fsa`,
etc, using StaticArrays as a backend.

The type definitions are not "perfect" matches because the type parameters are
different. However, it should cover common method signatures and constructors.
"""
module FixedSizeArraysWillBeRemoved

using ..StaticArrays

const FixedSizeArrays = FixedSizeArraysWillBeRemoved
export FixedSizeArrays # Ensure deprecated module name is in scope after import

export FixedArray
export FixedVector
export FixedMatrix
export Mat
export Vec
export Point
export @fsa
export FixedVectorNoTuple

const FixedArray = StaticArray
const FixedVector = StaticVector
const FixedMatrix = StaticMatrix
const Mat = SMatrix
const FixedVectorNoTuple = FieldVector

function fsa_ast(ex)
    @assert isa(ex, Expr)
    if ex.head == :vect # Vector
        return Expr(:call, SVector{length(ex.args)}, Expr(:tuple, ex.args...))
    elseif ex.head == :hcat # 1 x n
        s1 = 1
        s2 = length(ex.args)
        return Expr(:call, SMatrix{s1, s2}, Expr(:tuple, ex.args...))
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
            return Expr(:call, SMatrix{s1, s2}, Expr(:tuple, exprs...))
        else # n x 1
            return Expr(:call, SMatrix{length(ex.args), 1}, Expr(:tuple, ex.args...))
        end
    end
end
macro fsa(ex)
    expr = fsa_ast(ex)
    esc(expr)
end


function Base.isnan(x::StaticArray)
    for elem in x
        isnan(elem) && return true
    end
    false
end



function unit(::Type{T}, i::Integer) where T <: StaticVector
    T(ntuple(Val(length(T))) do j
        ifelse(i == j, 1, 0)
    end)
end

export unit

function Base.extrema(a::AbstractVector{T}) where T <: StaticVector
    ET = eltype(T)
    reduce((x, v)-> (min.(x[1], v), max.(x[2], v)), a; init = (T(typemax(ET)), T(typemin(ET))))
end
function Base.minimum(a::AbstractVector{T}) where T <: StaticVector
    reduce((x, v)-> min.(x[1], v), a; init=T(typemax(eltype(T))))
end
function Base.maximum(a::AbstractVector{T}) where T <: StaticVector
    reduce((x, v)-> max.(x[1], v), a; init=T(typemin(eltype(T))))
end



macro fixed_vector(name, parent)
    esc(quote
        struct $(name){S, T} <: $(parent){S, T}
            data::NTuple{S, T}

            function $(name){S, T}(x::NTuple{S,T}) where {S, T}
                new{S, T}(x)
            end

            function $(name){S, T}(x::NTuple{S,Any}) where {S, T}
                new{S, T}(StaticArrays.convert_ntuple(T, x))
            end
        end
        size_or(::Type{$(name)}, or) = or
        eltype_or(::Type{$(name)}, or) = or
        eltype_or(::Type{$(name){S, T} where S}, or) where {T} = T
        eltype_or(::Type{$(name){S, T} where T}, or) where {S} = or
        eltype_or(::Type{$(name){S, T}}, or) where {S, T} = T

        size_or(::Type{$(name){S, T} where S}, or) where {T} = or
        size_or(::Type{$(name){S, T} where T}, or) where {S} = Size{(S,)}()
        size_or(::Type{$(name){S, T}}, or) where {S, T} = (S,)
        # Array constructor
        @inline function $(name){S}(x::AbstractVector{T}) where {S, T}
            @assert S <= length(x)
            $(name){S, T}(ntuple(i-> x[i], Val(S)))
        end
        @inline function $(name){S, T1}(x::AbstractVector{T2}) where {S, T1, T2}
            @assert S <= length(x)
            $(name){S, T1}(ntuple(i-> T1(x[i]), Val(S)))
        end

        @inline function $(name){S, T}(x) where {S, T}
            $(name){S, T}(ntuple(i-> T(x), Val(S)))
        end


        @inline function $(name){S}(x::T) where {S, T}
            $(name){S, T}(ntuple(i-> x, Val(S)))
        end
        @inline function $(name){1, T}(x::T) where T
            $(name){1, T}((x,))
        end
        @inline $(name)(x::NTuple{S}) where {S} = $(name){S}(x)
        @inline $(name)(x::T) where {S, T <: Tuple{Vararg{Any, S}}} = $(name){S, StaticArrays.promote_tuple_eltype(T)}(x)
        @inline function $(name){S}(x::T) where {S, T <: Tuple}
            $(name){S, StaticArrays.promote_tuple_eltype(T)}(x)
        end
        $(name){S, T}(x::StaticVector) where {S, T} = $(name){S, T}(Tuple(x))
        @generated function (::Type{$(name){S, T}})(x::$(name)) where {S, T}
            idx = [:(x[$i]) for i = 1:S]
            quote
                $($(name)){S, T}($(idx...))
            end
        end
        @generated function convert(::Type{$(name){S, T}}, x::$(name)) where {S, T}
            idx = [:(x[$i]) for i = 1:S]
            quote
                $($(name)){S, T}($(idx...))
            end
        end
        @generated function (::Type{SV})(x::StaticVector) where SV <: $(name)
            len = size_or(SV, size(x))[1]
            if length(x) == len
                :(SV(Tuple(x)))
            elseif length(x) > len
                elems = [:(x[$i]) for i = 1:len]
                :(SV($(Expr(:tuple, elems...))))
            else
                error("Static Vector too short: $x, target type: $SV")
            end
        end

        Base.@pure StaticArrays.Size(::Type{$(name){S, Any}}) where {S} = Size(S)
        Base.@pure StaticArrays.Size(::Type{$(name){S, T}}) where {S,T} = Size(S)

        Base.@propagate_inbounds function Base.getindex(v::$(name){S, T}, i::Int) where {S, T}
            v.data[i]
        end
        @inline Base.Tuple(v::$(name)) = v.data
        @inline Base.convert(::Type{$(name){S, T}}, x::NTuple{S, T}) where {S, T} = $(name){S, T}(x)
        @inline function Base.convert(::Type{$(name){S, T}}, x::Tuple) where {S, T}
            $(name){S, T}(convert(NTuple{S, T}, x))
        end

        @generated function StaticArrays.similar_type(::Type{SV}, ::Type{T}, s::Size{S}) where {SV <: $(name), T, S}
            if length(S) === 1
                $(name){S[1], T}
            else
                StaticArrays.default_similar_type(T,s(),Val{length(S)})
            end
        end

    end)
end


@fixed_vector Vec StaticVector
@fixed_vector Point StaticVector

end

Base.@deprecate_binding FixedSizeArrays FixedSizeArraysWillBeRemoved #=
    =# false ".  Use StaticArrays directly."
