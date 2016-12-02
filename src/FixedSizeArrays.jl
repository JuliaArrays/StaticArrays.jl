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
export Mat, Vec, Point
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

end
