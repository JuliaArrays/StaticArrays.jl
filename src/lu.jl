# define our own LU type, since LinearAlgebra.LU requires p::Vector
struct LU{L,U,p}
    L::L
    U::U
    p::p
end

# iteration for destructuring into components
Base.iterate(S::LU) = (S.L, Val(:U))
Base.iterate(S::LU, ::Val{:U}) = (S.U, Val(:p))
Base.iterate(S::LU, ::Val{:p}) = (S.p, Val(:done))
Base.iterate(S::LU, ::Val{:done}) = nothing

# LU decomposition
function lu(A::StaticMatrix, pivot::Union{Val{false},Val{true}}=Val(true))
    L, U, p = _lu(A, pivot)
    LU(L, U, p)
end

# For the square version, return explicit lower and upper triangular matrices.
# We would do this for the rectangular case too, but Base doesn't support that.
function lu(A::StaticMatrix{N,N}, pivot::Union{Val{false},Val{true}}=Val(true)) where {N}
    L, U, p = _lu(A, pivot)
    LU(LowerTriangular(L), UpperTriangular(U), p)
end

@generated function _lu(A::StaticMatrix{M,N,T}, pivot) where {M,N,T}
    if M*N ≤ 14*14
        :(__lu(A, pivot))
    else
        quote
            # call through to Base to avoid excessive time spent on type inference for large matrices
            f = lu(Matrix(A), pivot; check = false)
            # Trick to get the output eltype - can't rely on the result of f.L as
            # it's not type inferrable.
            T2 = arithmetic_closure(T)
            L = similar_type(A, T2, Size($M, $(min(M,N))))(f.L)
            U = similar_type(A, T2, Size($(min(M,N)), $N))(f.U)
            p = similar_type(A, Int, Size($M))(f.p)
            (L,U,p)
        end
    end
end

__lu(A::StaticMatrix{0,0,T}, ::Val{Pivot}) where {T,Pivot} =
    (SMatrix{0,0,typeof(one(T))}(), A, SVector{0,Int}())

__lu(A::StaticMatrix{0,1,T}, ::Val{Pivot}) where {T,Pivot} =
    (SMatrix{0,0,typeof(one(T))}(), A, SVector{0,Int}())

__lu(A::StaticMatrix{0,N,T}, ::Val{Pivot}) where {T,N,Pivot} =
    (SMatrix{0,0,typeof(one(T))}(), A, SVector{0,Int}())

__lu(A::StaticMatrix{1,0,T}, ::Val{Pivot}) where {T,Pivot} =
    (SMatrix{1,0,typeof(one(T))}(), SMatrix{0,0,T}(), SVector{1,Int}(1))

__lu(A::StaticMatrix{M,0,T}, ::Val{Pivot}) where {T,M,Pivot} =
    (SMatrix{M,0,typeof(one(T))}(), SMatrix{0,0,T}(), SVector{M,Int}(1:M))

__lu(A::StaticMatrix{1,1,T}, ::Val{Pivot}) where {T,Pivot} =
    (SMatrix{1,1}(one(T)), A, SVector(1))

__lu(A::StaticMatrix{1,N,T}, ::Val{Pivot}) where {N,T,Pivot} =
    (SMatrix{1,1,T}(one(T)), A, SVector{1,Int}(1))

function __lu(A::StaticMatrix{M,1}, ::Val{Pivot}) where {M,Pivot}
    @inbounds begin
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
            Ls = zeros(typeof(Ls))
        end
        L = [SVector{1}(one(eltype(Ls))); Ls]
        p = [SVector{1,Int}(kp); ps]
    end
    return (SMatrix{M,1}(L), U, p)
end

function __lu(A::StaticMatrix{M,N,T}, ::Val{Pivot}) where {M,N,T,Pivot}
    @inbounds begin
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
            Ls = zeros(typeof(Ls))
        end

        # Update the rest
        Arest = A[ps,tailindices(Val{N})] - Ls*Ufirst[:,tailindices(Val{N})]
        Lrest, Urest, prest = __lu(Arest, Val(Pivot))
        p = [SVector{1,Int}(kp); ps[prest]]
        L = [[SVector{1}(one(eltype(Ls))); Ls[prest]] [zeros(typeof(SMatrix{1}(Lrest[1,:]))); Lrest]]
        U = [Ufirst; [zeros(typeof(Urest[:,1])) Urest]]
    end
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

\(F::LU, v::AbstractVector) = F.U \ (F.L \ v[F.p])
\(F::LU, B::AbstractMatrix) = F.U \ (F.L \ B[F.p,:])
