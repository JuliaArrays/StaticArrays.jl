"""
Compatibility layer for transferring from FixedSizeArrays. This provides
alternative definitions of `Vec`, `Mat`, `Point`, `FixedVectorNoTuple`, `@fsa`,
etc, using StaticArrays as a backend.

The type definitions are not "perfect" matches because the type parameters are
different. However, it should cover common method signatures and constructors.
"""
module FixedSizeArrays

using ..StaticArrays

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



function unit{T <: StaticVector}(::Type{T}, i::Integer)
    T(ntuple(Val{length(T)}) do j
        ifelse(i == j, 1, 0)
    end)
end

export unit

function Base.extrema{T <: StaticVector}(a::AbstractVector{T})
    reduce((x, v)-> (min.(x[1], v), max.(x[2], v)), a)
end
function Base.minimum{T <: StaticVector}(a::AbstractVector{T})
    reduce((x, v)-> min.(x[1], v), a)
end
function Base.maximum{T <: StaticVector}(a::AbstractVector{T})
    reduce((x, v)-> max.(x[1], v), a)
end



macro fixed_vector(name, parent)
    esc(quote
        immutable $(name){S, T} <: $(parent){S, T}
            data::NTuple{S, T}

            function (::Type{$(name){S, T}}){S, T}(x::NTuple{S,T})
                new{S, T}(x)
            end

            function (::Type{$(name){S, T}}){S, T}(x::NTuple{S,Any})
                new{S, T}(StaticArrays.convert_ntuple(T, x))
            end
        end
        # Array constructor
        @inline function (::Type{$(name){S}}){S, T}(x::AbstractVector{T})
            @assert S <= length(x)
            $(name){S, T}(ntuple(i-> x[i], Val{S}))
        end
        @inline function (::Type{$(name){S, T1}}){S, T1, T2}(x::AbstractVector{T2})
            @assert S <= length(x)
            $(name){S, T1}(ntuple(i-> T1(x[i]), Val{S}))
        end

        @inline function (::Type{$(name){S, T}}){S, T}(x)
            $(name){S, T}(ntuple(i-> T(x), Val{S}))
        end


        @inline function (::Type{$(name){S}}){S, T}(x::T)
            $(name){S, T}(ntuple(i-> x, Val{S}))
        end
        @inline function (::Type{$(name){1, T}}){T}(x::T)
            $(name){1, T}((x,))
        end
        @inline (::Type{$(name)}){S}(x::NTuple{S}) = $(name){S}(x)
        @inline function (::Type{$(name){S}}){S, T <: Tuple}(x::T)
            $(name){S, StaticArrays.promote_tuple_eltype(T)}(x)
        end
        (::Type{$(name){S, T}}){S, T}(x::StaticVector) = $(name){S, T}(Tuple(x))
        @generated function (::Type{$(name){S, T}}){S, T}(x::$(name))
            idx = [:(x[$i]) for i = 1:S]
            quote
                $($(name)){S, T}($(idx...))
            end
        end
        @generated function convert{S, T}(::Type{$(name){S, T}}, x::$(name))
            idx = [:(x[$i]) for i = 1:S]
            quote
                $($(name)){S, T}($(idx...))
            end
        end
        @generated function (::Type{SV}){SV <: $(name)}(x::StaticVector)
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

        Base.@pure StaticArrays.Size{S}(::Type{$(name){S, Any}}) = Size(S)
        Base.@pure StaticArrays.Size{S,T}(::Type{$(name){S, T}}) = Size(S)

        Base.@propagate_inbounds function Base.getindex{S, T}(v::$(name){S, T}, i::Int)
            v.data[i]
        end
        @inline Base.Tuple(v::$(name)) = v.data
        @inline Base.convert{S, T}(::Type{$(name){S, T}}, x::NTuple{S, T}) = $(name){S, T}(x)
        @inline Base.convert(::Type{$(name)}, x::StaticVector) = SV(x)
        @inline function Base.convert{S, T}(::Type{$(name){S, T}}, x::Tuple)
            $(name){S, T}(convert(NTuple{S, T}, x))
        end

        @generated function StaticArrays.similar_type{SV <: $(name), T, S}(::Type{SV}, ::Type{T}, s::Size{S})
            if length(S) === 1
                $(name){S[1], T}
            else
                StaticArrays.default_similar_type(T,s(),Val{length(S)})
            end
        end
        size_or(::Type{$(name)}, or) = or
        eltype_or(::Type{$(name)}, or) = or
        eltype_or{T}(::Type{$(name){S, T} where S}, or) = T
        eltype_or{S}(::Type{$(name){S, T} where T}, or) = or
        eltype_or{S, T}(::Type{$(name){S, T}}, or) = T

        size_or{T}(::Type{$(name){S, T} where S}, or) = or
        size_or{S}(::Type{$(name){S, T} where T}, or) = Size{(S,)}()
        size_or{S, T}(::Type{$(name){S, T}}, or) = (S,)
    end)
end


@fixed_vector Vec StaticVector
@fixed_vector Point StaticVector

end
