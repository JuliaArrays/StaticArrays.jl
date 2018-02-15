# LU decomposition
function lu(A::StaticMatrix, pivot::Union{Type{Val{false}},Type{Val{true}}}=Val{true})
    L,U,p = _lu(A, pivot)
    (L,U,p)
end

# For the square version, return explicit lower and upper triangular matrices.
# We would do this for the rectangular case too, but Base doesn't support that.
function lu(A::StaticMatrix{N,N}, pivot::Union{Type{Val{false}},Type{Val{true}}}=Val{true}) where {N}
    L,U,p = _lu(A, pivot)
    (LowerTriangular(L), UpperTriangular(U), p)
end

_lu(A::StaticMatrix{0,0,T}, ::Type{Val{Pivot}}) where {T,Pivot} =
    (SMatrix{0,0,typeof(one(T))}(), A, SVector{0,Int}())

_lu(A::StaticMatrix{0,1,T}, ::Type{Val{Pivot}}) where {T,Pivot} =
    (SMatrix{0,0,typeof(one(T))}(), A, SVector{0,Int}())

_lu(A::StaticMatrix{0,N,T}, ::Type{Val{Pivot}}) where {T,N,Pivot} =
    (SMatrix{0,0,typeof(one(T))}(), A, SVector{0,Int}())

_lu(A::StaticMatrix{1,0,T}, ::Type{Val{Pivot}}) where {T,Pivot} =
    (SMatrix{1,0,typeof(one(T))}(), SMatrix{0,0,T}(), SVector{1,Int}(1))

_lu(A::StaticMatrix{M,0,T}, ::Type{Val{Pivot}}) where {T,M,Pivot} =
    (SMatrix{M,0,typeof(one(T))}(), SMatrix{0,0,T}(), SVector{M,Int}(1:M))

_lu(A::StaticMatrix{1,1,T}, ::Type{Val{Pivot}}) where {T,Pivot} =
    (SMatrix{1,1}(one(T)), A, SVector(1))

_lu(A::StaticMatrix{1,N,T}, ::Type{Val{Pivot}}) where {N,T,Pivot} =
    (SMatrix{1,1,T}(one(T)), A, SVector{1,Int}(1))

function _lu(A::StaticMatrix{M,1}, ::Type{Val{Pivot}}) where {M,Pivot}
    kp = 1
    if Pivot
        amax = abs(A[1,1])
        for i = 2:M
            absi = abs(A[i,1])
            if absi > amax
                kp = i
                amax = absi
            end
        end
    end
    ps = tailindices(Val{M})
    if kp != 1
        ps = setindex(ps, 1, kp-1)
    end
    U = SMatrix{1,1}(A[kp,1])
    # Scale first column
    Akkinv = inv(A[kp,1])
    Ls = A[ps,1] * Akkinv
    if !isfinite(Akkinv)
        Ls = zeros(Ls)
    end
    L = [SVector{1}(one(eltype(Ls))); Ls]
    p = [SVector{1,Int}(kp); ps]
    return (SMatrix{M,1}(L), U, p)
end

function _lu(A::StaticMatrix{M,N,T}, ::Type{Val{Pivot}}) where {M,N,T,Pivot}
    kp = 1
    if Pivot
        amax = abs(A[1,1])
        for i = 2:M
            absi = abs(A[i,1])
            if absi > amax
                kp = i
                amax = absi
            end
        end
    end
    ps = tailindices(Val{M})
    if kp != 1
        ps = setindex(ps, 1, kp-1)
    end
    Ufirst = SMatrix{1,N}(A[kp,:])
    # Scale first column
    Akkinv = inv(A[kp,1])
    Ls = A[ps,1] * Akkinv
    if !isfinite(Akkinv)
        Ls = zeros(Ls)
    end

    # Update the rest
    Arest = A[ps,tailindices(Val{N})] - Ls*Ufirst[:,tailindices(Val{N})]
    Lrest, Urest, prest = _lu(Arest, Val{Pivot})
    p = [SVector{1,Int}(kp); ps[prest]]
    L = [[SVector{1}(one(eltype(Ls))); Ls[prest]] [zeros(SMatrix{1}(Lrest[1,:])); Lrest]]
    U = [Ufirst; [zeros(Urest[:,1]) Urest]]

    return (L, U, p)
end

# Create SVector(2,3,...,M)
# Note that
#     tailindices(::Type{Val{M}}) where {M} = SVector(Base.tail(ntuple(identity, Val{M})))
# works, too, but is only inferrable for M ≤ 14 (at least up to Julia 0.7.0-DEV.4021)
@generated function tailindices(::Type{Val{M}}) where {M}
    :(SVector{$(M-1),Int}($(tuple(2:M...))))
end

# Base.lufact() interface is fairly inherently type unstable.  Punt on
# implementing that, for now...
