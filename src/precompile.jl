@setup_workload begin
    @compile_workload begin
        for T in (Float32, Float64, Int)   # some common eltypes
            for S = 2:4                    # some common sizes
                L = S*S
                x = fill(0., SMatrix{S, S, T, L})
                x * x
                y = fill(0., SVector{S, T})
                x * y
                inv(x)
                transpose(x)
                _adjoint(Size(S, S), x)
                # @assert precompile(Tuple{Core.kwftype(typeof(minimum)),NamedTuple{(:dims,), Tuple{Int}},typeof(minimum),SMatrix{S, S, T, L}})
                # @assert precompile(Tuple{Core.kwftype(typeof(maximum)),NamedTuple{(:dims,), Tuple{Int}},typeof(maximum),SMatrix{S, S, T, L}})
                getindex(x, SOneTo(S-1), SOneTo(S-1))
            end
        end

        # Some expensive generators
        # @assert precompile(Tuple{typeof(which(__broadcast,(Any,Size,Tuple{Vararg{Size}},Vararg{Any},)).generator.gen),Any,Any,Any,Any,Any,Any})
        # @assert precompile(Tuple{typeof(which(_zeros,(Size,Type{<:StaticArray},)).generator.gen),Any,Any,Any,Type,Any})
        # @assert precompile(Tuple{typeof(which(_mapfoldl,(Any,Any,Colon,Any,Size,Vararg{StaticArray},)).generator.gen),Any,Any,Any,Any,Any,Any,Any,Any})

        # broadcast_getindex
        for m = 0:5, n = m:5
            broadcast_getindex(Tuple(1:m), 1, CartesianIndex(n))
        end
    end
end
