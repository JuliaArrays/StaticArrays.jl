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

@inline function Base.getproperty(F::LU, s::Symbol)
    if s === :P
        U = getfield(F, :U)
        p = getfield(F, :p)
        one(similar_type(p, Size(U)))[:,invperm(p)]
    else
        getfield(F, s)
    end
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::LU)
    println(io, LU) # Don't show full type - this will be in the factors
    println(io, "L factor:")
    show(io, mime, F.L)
    println(io, "\nU factor:")
    show(io, mime, F.U)
end

const StaticLUMatrix{N,M,T} = Union{StaticMatrix{N,M,T}, Symmetric{T,<:StaticMatrix{N,M,T}}, Hermitian{T,<:StaticMatrix{N,M,T}}}

# LU decomposition
pivot_options = if isdefined(LinearAlgebra, :PivotingStrategy) # introduced in Julia v1.7
    (:(Val{true}), :(Val{false}), :NoPivot, :RowMaximum)
else
    (:(Val{true}), :(Val{false}))
end
for pv in pivot_options
    # ... define each `pivot::Val{true/false}` method individually to avoid ambiguities
    @eval function lu(A::StaticLUMatrix, pivot::$pv; check = true)
        L, U, p = _lu(A, pivot, check)
        LU(L, U, p)
    end

    # For the square version, return explicit lower and upper triangular matrices.
    # We would do this for the rectangular case too, but Base doesn't support that.
    @eval function lu(A::StaticLUMatrix{N,N}, pivot::$pv; check = true) where {N}
        L, U, p = _lu(A, pivot, check)
        LU(LowerTriangular(L), UpperTriangular(U), p)
    end
end
lu(A::StaticLUMatrix; check = true) = lu(A, Val(true); check=check)

# location of the first zero on the diagonal, 0 when not found
function _first_zero_on_diagonal(A::StaticLUMatrix{M,N,T}) where {M,N,T}
    if @generated
        quote
            $(map(i -> :(A[$i, $i] == zero(T) && return $i), 1:min(M, N))...)
            0
        end
    else
        for i in 1:min(M, N)
            A[i, i] == 0 && return i
        end
        0
    end
end

issuccess(F::LU) = _first_zero_on_diagonal(F.U) == 0

@generated function _lu(A::StaticLUMatrix{M,N,T}, pivot, check) where {M,N,T}
    if M*N ≤ 14*14
        _pivot = if isdefined(LinearAlgebra, :PivotingStrategy) # v1.7 feature
            pivot === RowMaximum ? Val(true) : pivot === NoPivot ? Val(false) : pivot()
        else
            pivot()
        end
        quote
            L, U, P = __lu(A, $(_pivot))
            if check
                i = _first_zero_on_diagonal(U)
                i == 0 || throw(SingularException(i))
            end
            L, U, P
        end
    else
        _pivot = if isdefined(LinearAlgebra, :PivotingStrategy) # v1.7 feature
            pivot === Val{true} ? RowMaximum() : pivot === Val{false} ? NoPivot() : pivot()
        else
            pivot()
        end
        quote
            # call through to Base to avoid excessive time spent on type inference for large matrices
            f = lu(Matrix(A), $(_pivot); check = check)
            # Trick to get the output eltype - can't rely on the result of f.L as
            # it's not type inferable.
            T2 = arithmetic_closure(T)
            L = similar_type(A, T2, Size($M, $(min(M,N))))(f.L)
            U = similar_type(A, T2, Size($(min(M,N)), $N))(f.U)
            p = similar_type(A, Int, Size($M))(f.p)
            L, U, p
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

__lu(A::LinearAlgebra.HermOrSym{T,<:StaticMatrix{1,1,T}}, ::Val{Pivot}) where {T,Pivot} =
    (SMatrix{1,1}(one(T)), A.data, SVector(1))

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

function __lu(A::StaticLUMatrix{M,N,T}, ::Val{Pivot}) where {M,N,T,Pivot}
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
# works, too, but is only inferable for M ≤ 14 (at least up to Julia 0.7.0-DEV.4021)
@generated function tailindices(::Type{Val{M}}) where {M}
    :(SVector{$(M-1),Int}($(tuple(2:M...))))
end

\(F::LU, v::AbstractVector) = F.U \ (F.L \ v[F.p])
\(F::LU, B::AbstractMatrix) = F.U \ (F.L \ B[F.p,:])

/(B::AbstractMatrix, F::LU) = @inbounds ((B/F.U)/F.L)[:,invperm(F.p)]
