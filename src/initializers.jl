"""
    SA[ elements ]
    SA{T}[ elements ]

Create `SArray` literals using array construction syntax. The element type is
inferred by promoting `elements` to a common type or set to `T` when `T` is
provided explicitly.

# Examples:

* `SA[1.0, 2.0]` creates a length-2 `SVector` of `Float64` elements.
* `SA[1 2; 3 4]` creates a 2×2 `SMatrix` of `Int`s.
* `SA[1 2]` creates a 1×2 `SMatrix` of `Int`s.
* `SA{Float32}[1, 2]` creates a length-2 `SVector` of `Float32` elements.

A couple of helpful type aliases are also provided:

* `SA_F64[1, 2]` creates a length-2 `SVector` of `Float64` elements
* `SA_F32[1, 2]` creates a length-2 `SVector` of `Float32` elements
"""
struct SA{T} ; end

const SA_F32 = SA{Float32}
const SA_F64 = SA{Float64}

@inline similar_type(::Type{SA}, ::Size{S}) where {S} = SArray{Tuple{S...}}
@inline similar_type(::Type{SA{T}}, ::Size{S}) where {T,S} = SArray{Tuple{S...}, T}

# These definitions are duplicated to avoid matching `sa === Union{}` in the
# neater-looking alternative `sa::Type{<:SA}`.
@inline Base.getindex(sa::Type{SA}, xs...)            = similar_type(sa, Size(length(xs)))(xs)
@inline Base.getindex(sa::Type{SA{T}}, xs...) where T = similar_type(sa, Size(length(xs)))(xs)

@inline Base.typed_vcat(sa::Type{SA}, xs::Number...)            = similar_type(sa, Size(length(xs)))(xs)
@inline Base.typed_vcat(sa::Type{SA{T}}, xs::Number...) where T = similar_type(sa, Size(length(xs)))(xs)

@inline Base.typed_hcat(sa::Type{SA}, xs::Number...)            = similar_type(sa, Size(1,length(xs)))(xs)
@inline Base.typed_hcat(sa::Type{SA{T}}, xs::Number...) where T = similar_type(sa, Size(1,length(xs)))(xs)

Base.@pure function _SA_hvcat_transposed_size(rows)
    M = rows[1]
    if any(r->r != M, rows)
        # @pure may not throw... probably. See
        # https://discourse.julialang.org/t/can-pure-functions-throw-an-error/18459
        return nothing
    end
    Size(M, length(rows))
end

@inline function _SA_typed_hvcat(sa, rows, xs)
    msize = _SA_hvcat_transposed_size(rows)
    if msize === nothing
        throw(ArgumentError("SA[...] matrix rows of length $rows are inconsistent"))
    end
    # hvcat lowering is row major ordering, so we must transpose
    transpose(similar_type(sa, msize)(xs))
end

@inline Base.typed_hvcat(sa::Type{SA}, rows::Dims, xs::Number...)            = _SA_typed_hvcat(sa, rows, xs)
@inline Base.typed_hvcat(sa::Type{SA{T}}, rows::Dims, xs::Number...) where T = _SA_typed_hvcat(sa, rows, xs)

if VERSION >= v"1.7"
@generated function reorder(x::NTuple{M,Any}, ::Val{N1}, ::Val{N2}) where {M,N1,N2}
    a = Base.PermutedDimsArray(reshape(1:M, N1, N2, :), (2, 1, 3))
    args = (:(x[$i]) for i in a)
    return :(tuple($(args...)))
end
    
@inline function _SA_typed_hvncat(sa, dimsshape, row_first, xs)
    msize = Size(dimsshape)
    xs′ = row_first ? reorder(xs, Val(msize[1]), Val(msize[2])) : xs
    x = similar_type(sa, msize)(xs′)
end

@inline Base.typed_hvncat(sa::Type{SA}, dimsshape::Dims, row_first::Bool, xs::Number...) = _SA_typed_hvncat(sa, dimsshape, row_first, xs)
@inline Base.typed_hvncat(sa::Type{SA{T}}, dimsshape::Dims, row_first::Bool, xs::Number...) where T = _SA_typed_hvncat(sa, dimsshape, row_first, xs)

@inline function _SA_typed_hvncat(sa, ::Val{dim}, xs) where {dim}
    msize = Size(ntuple(_->1, Val(dim))..., length(xs))
    x = similar_type(sa, msize)(xs)
end

@inline Base.typed_hvncat(sa::Type{SA}, dim::Int, xs::Number...) = _SA_typed_hvncat(sa, Val(dim-1), xs)
@inline Base.typed_hvncat(sa::Type{SA{T}}, dim::Int, xs::Number...) where T = _SA_typed_hvncat(sa, Val(dim-1), xs)
end
