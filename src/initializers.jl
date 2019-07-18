"""
    SA[ array initializer ]

A type for initializing static array literals using array construction syntax.
Returns an `SVector` or `SMatrix`.

# Examples:

* `SA[x, y]` creates a length-2 SVector
* `SA[a b; c d]` creates a 2×2 SMatrix
* `SA[a b]` creates a 1×2 SMatrix
"""
struct SA ; end

Base.getindex(::Type{SA}, xs...) = SVector(xs)
Base.typed_vcat(::Type{SA}, xs::Number...) = SVector(xs)
Base.typed_hcat(::Type{SA}, xs::Number...) = SMatrix{1,length(xs)}(xs)

Base.@pure function _SA_hvcat_transposed_type(rows)
    M = rows[1]
    if any(r->r != M, rows)
        # @pure may not throw... probably. See
        # https://discourse.julialang.org/t/can-pure-functions-throw-an-error/18459
        return nothing
    end
    SMatrix{M,length(rows)}
end

@inline function Base.typed_hvcat(::Type{SA}, rows::Dims, xs::Number...)
    mtype = _SA_hvcat_transposed_type(rows)
    if mtype === nothing
        throw(ArgumentError("SA[...] matrix rows of length $rows are inconsistent"))
    end
    # hvcat lowering is row major ordering, so must transpose
    transpose(mtype(xs))
end


