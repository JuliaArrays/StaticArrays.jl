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
                minimum(x; dims=1)
                minimum(x; dims=2)
                maximum(x; dims=1)
                maximum(x; dims=2)
                getindex(x, SOneTo(S-1), SOneTo(S-1))
                y .* x .* y'
                zero(y)
                zero(x)
            end
        end

        # broadcast_getindex
        for m = 0:5, n = m:5
            broadcast_getindex(Tuple(1:m), 1, CartesianIndex(n))
        end
    end
end
