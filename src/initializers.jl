"""
    SA[ elements ]
    SA{T}[ elements ]

Create `SArray` literals using array construction syntax. The element type is
inferred by promoting `elements` to a common type or set to `T` when `T` is
provided explicitly.

# Examples:

* `SA[1.0, 2.0]` creates a length-2 `SVector` of `Float64` elements.
* `SA[1 2; 3 4]` creates a 2×2 SMatrix of `Int`s.
* `SA[1 2]` creates a 1×2 SMatrix of `Int`s.
* `SA{Float32}[1, 2]` creates a length-2 `SVector` of `Float32` elements.
"""
struct SA{T} ; end

@inline similar_type(::Type{SA}, ::Size{S}) where {S} = SArray{Tuple{S...}}
@inline similar_type(::Type{SA{T}}, ::Size{S}) where {T,S} = SArray{Tuple{S...}, T}

Base.@pure _SA_type(sa::Type{SA}, len::Int) = SVector{len}
Base.@pure _SA_type(sa::Type{SA{T}}, len::Int) where {T} = SVector{len,T}

@inline Base.getindex(sa::Type{<:SA}, xs...) where T = similar_type(sa, Size(length(xs)))(xs)
@inline Base.typed_vcat(sa::Type{<:SA}, xs::Number...) where T = similar_type(sa, Size(length(xs)))(xs)
@inline Base.typed_hcat(sa::Type{<:SA}, xs::Number...) where T = similar_type(sa, Size(1,length(xs)))(xs)

Base.@pure function _SA_hvcat_transposed_size(rows)
    M = rows[1]
    if any(r->r != M, rows)
        # @pure may not throw... probably. See
        # https://discourse.julialang.org/t/can-pure-functions-throw-an-error/18459
        return nothing
    end
    Size(M, length(rows))
end

@inline function Base.typed_hvcat(sa::Type{<:SA}, rows::Dims, xs::Number...) where T
    msize = _SA_hvcat_transposed_size(rows)
    if msize === nothing
        throw(ArgumentError("SA[...] matrix rows of length $rows are inconsistent"))
    end
    # hvcat lowering is row major ordering, so we must transpose
    transpose(similar_type(sa, msize)(xs))
end

