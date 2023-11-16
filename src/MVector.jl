#####################
## MVector methods ##
#####################
"""
    @MVector [a, b, c, d]
    @MVector [i for i in 1:2]
    @MVector ones(2)

A convenience macro to construct `MVector`.
See [`@SArray`](@ref) for detailed features.
"""
macro MVector(ex)
    static_vector_gen(MVector, ex, __module__)
end

# Named field access for the first four elements, using the conventional field
# names from low-dimensional geometry and computer graphics:
# - (x, y, z, w) where w is a homogenous coordinate.
# - (r, g, b, a) where a is an alpha component.
# - (xy, rg, xyz, ...) to obtain a subset of the vector via swizzling.

@inline function swizzle(v::SVector, index, indices...)
    isempty(indices) && return v[index]
    indices = (index, indices...)
    SVector(ntuple(i -> v[indices[i]], length(indices)))
end
@inline function swizzle(v::MVector, index, indices...)
    isempty(indices) && return v[index]
    indices = (index, indices...)
    MVector(ntuple(i -> v[indices[i]], length(indices)))
end

@inline function swizzle!(v::MVector, value, indices...)
    for (i, index) in enumerate(indices)
        setindex!(v, value[i], index)
    end
    value
end

let dimension_names = zip((:x, :y, :z, :w), (:r, :g, :b, :a))
    getproperty_bodies = [Expr[], Expr[], Expr[], Expr[]]
    setproperty_bodies = [Expr[], Expr[], Expr[], Expr[]]
    for (i, (dx1, dr1)) in enumerate(dimension_names)
        field1 = dx1
        field2 = dr1
        push!(getproperty_bodies[i], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle(v, $i)))
        push!(setproperty_bodies[i], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle!(v, value, $i)))
        for (j, (dx2, dr2)) in enumerate(dimension_names)
            field1 = Symbol(dx1, dx2)
            field2 = Symbol(dr1, dr2)
            push!(getproperty_bodies[max(i, j)], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle(v, $i, $j)))
            j ≠ i && push!(setproperty_bodies[max(i, j)], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle!(v, value, $i, $j)))
            for (k, (dx3, dr3)) in enumerate(dimension_names)
                field1 = Symbol(dx1, dx2, dx3)
                field2 = Symbol(dr1, dr2, dr3)
                push!(getproperty_bodies[max(i, j, k)], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle(v, $i, $j, $k)))
                (k ≠ j && k ≠ i) && push!(setproperty_bodies[max(i, j, k)], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle!(v, value, $i, $j, $k)))
                for (l, (dx4, dr4)) in enumerate(dimension_names)
                    field1 = Symbol(dx1, dx2, dx3, dx4)
                    field2 = Symbol(dr1, dr2, dr3, dr4)
                    push!(getproperty_bodies[max(i, j, k, l)], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle(v, $i, $j, $k, $l)))
                    (l ≠ k && l ≠ j && l ≠ i) && push!(setproperty_bodies[max(i, j, k, l)], :((name === $(QuoteNode(field1)) || name === $(QuoteNode(field2))) && return swizzle!(v, value, $i, $j, $k, $l)))
                end
            end
        end
    end
    for i in 1:4
        @eval function Base.getproperty(v::Union{SVector{$i},MVector{$i}},
                                                name::Symbol)
            $(foldl(append!, getproperty_bodies[1:i])...)
            getfield(v, name)
        end
        @eval function Base.setproperty!(v::MVector{$i}, name::Symbol, value)
            $(foldl(append!, setproperty_bodies[1:i])...)
            setfield!(v, name, value)
        end
    end
end
