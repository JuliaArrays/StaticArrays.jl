function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    # These make a difference for Makie's TTFP and should help others too
    # Total time for this to run is ~2s. To time it, disable:
    #   - the `_precompile_()` call in StaticArrays.jl
    #   - the jl_generating_output line above
    # and then do
    #     @time StaticArrays._precompile_()

    for T in (Float32, Float64, Int)   # some common eltypes
        for S = 2:4                    # some common sizes
            L = S*S
            @assert precompile(Tuple{typeof(*),SMatrix{S, S, T, L},SMatrix{S, S, T, L}})
            @assert precompile(Tuple{typeof(*),SMatrix{S, S, T, L},SVector{S, T}})
            @assert precompile(Tuple{typeof(inv),SMatrix{S, S, T, L}})
            @assert precompile(Tuple{typeof(transpose),SMatrix{S, S, T, L}})
            @assert precompile(Tuple{typeof(_adjoint),Size{(S, S)},SMatrix{S, S, T, L}})
            @assert precompile(Tuple{Core.kwftype(typeof(minimum)),NamedTuple{(:dims,), Tuple{Int}},typeof(minimum),SMatrix{S, S, T, L}})
            @assert precompile(Tuple{Core.kwftype(typeof(maximum)),NamedTuple{(:dims,), Tuple{Int}},typeof(maximum),SMatrix{S, S, T, L}})
            @assert precompile(Tuple{typeof(getindex),SMatrix{S, S, T, L},SOneTo{S-1},SOneTo{S-1}})
        end
    end

    # TODO: These fail to precompile on v1.11 pre-release
    if VERSION < v"1.11.0-0"
        # Some expensive generators
        @assert precompile(Tuple{typeof(which(__broadcast,(Any,Size,Tuple{Vararg{Size}},Vararg{Any},)).generator.gen),Any,Any,Any,Any,Any,Any})
        @assert precompile(Tuple{typeof(which(_zeros,(Size,Type{<:StaticArray},)).generator.gen),Any,Any,Any,Type,Any})
        @assert precompile(Tuple{typeof(which(_mapfoldl,(Any,Any,Colon,Any,Size,Vararg{StaticArray},)).generator.gen),Any,Any,Any,Any,Any,Any,Any,Any})
    end
    # broadcast_getindex
    for m = 0:5, n = m:5
        @assert precompile(Tuple{typeof(broadcast_getindex),NTuple{m,Int},Int,CartesianIndex{n}})
    end
end
