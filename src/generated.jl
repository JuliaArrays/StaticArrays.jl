#=
"""
     unroll_tuple(f, Val{N})

Like the `Base.ntuple` function, but generates fast code beyond N = 16.
"""
=#
@generated function unroll_tuple(f, ::Type{Val{N}}) where {N}
    @assert(N isa Int)
    exprs = [:(f($i)) for i = 1:N]
    quote
        @_propagate_inbounds_meta
        return $(Expr(:tuple, exprs...))
    end
end
