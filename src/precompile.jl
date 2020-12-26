function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    # These make a difference for Makie's TTFP and should help others too
    # Total time for this to run is ~2s. To time it, disable:
    #   - the `_precompile_()` call in StaticArrays.jl
    #   - the jl_generating_output line above
    # and then do
    #     @time StaticArrays._precompile_()

    for T in (Float32, Float64, Int)   # some common eltypes
        @assert precompile(Tuple{typeof(*),SMatrix{4, 4, T, 16},SMatrix{4, 4, T, 16}})
        @assert precompile(Tuple{typeof(*),SMatrix{3, 3, T,  9},SMatrix{3, 3, T,  9}})
        @assert precompile(Tuple{typeof(*),SMatrix{2, 2, T,  4},SMatrix{2, 2, T,  4}})
        @assert precompile(Tuple{typeof(*),SMatrix{2, 2, T,  4},SMatrix{2, 4, T,  8}})

        @assert precompile(Tuple{Core.kwftype(typeof(minimum)),NamedTuple{(:dims,), Tuple{Int}},typeof(minimum),SMatrix{2, 4, T, 8}})
        @assert precompile(Tuple{Core.kwftype(typeof(maximum)),NamedTuple{(:dims,), Tuple{Int}},typeof(maximum),SMatrix{2, 4, T, 8}})

        @assert precompile(Tuple{typeof(inv),SMatrix{4, 4, T, 16}})
        @assert precompile(Tuple{typeof(inv),SMatrix{3, 3, T,  9}})

        @assert precompile(Tuple{typeof(getindex),SMatrix{4, 4, T, 16},SOneTo{3},SOneTo{3}})

        @assert precompile(Tuple{typeof(copy),Broadcasted{StaticArrayStyle{2}, Tuple{SOneTo{2}, SOneTo{1}}, typeof(-), Tuple{SMatrix{2, 1, T, 2}, SMatrix{2, 1, T, 2}}}})

        @assert precompile(Tuple{typeof(transpose),SMatrix{3, 3, T, 9}})
        @assert precompile(Tuple{typeof(_adjoint),Size{(4, 2)},SMatrix{4, 2, T, 8}})
    end

    # Some expensive generators
    @assert precompile(Tuple{typeof(which(_broadcast,(Any,Size,Tuple{Vararg{Size}},Vararg{Any},)).generator.gen),Any,Any,Any,Any,Any,Any})
    @assert precompile(Tuple{typeof(which(_zeros,(Size,Type{<:StaticArray},)).generator.gen),Any,Any,Any,Type,Any})
    @assert precompile(Tuple{typeof(which(combine_sizes,(Tuple{Vararg{Size}},)).generator.gen),Any,Any})
    @assert precompile(Tuple{typeof(which(_mapfoldl,(Any,Any,Colon,Any,Size,Vararg{StaticArray},)).generator.gen),Any,Any,Any,Any,Any,Any,Any,Any})
end
