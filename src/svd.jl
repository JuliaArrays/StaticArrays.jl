# Singular Value Decomposition

# We need our own SVD factorization struct, as LinearAlgebra.SVD assumes
# Base.Vector for `S`, and that the `U` and `Vt` have the same
struct SVD{T,TU,TS,TVt} <: Factorization{T}
    U::TU
    S::TS
    Vt::TVt
end
SVD(U::AbstractArray{T}, S::AbstractVector, Vt::AbstractArray{T}) where {T} = SVD{T,typeof(U),typeof(S),typeof(Vt)}(U, S, Vt)

@inline function Base.getproperty(F::SVD, s::Symbol)
    if s === :V
        return getfield(F, :Vt)'
    else
        return getfield(F, s)
    end
end

# iteration for destructuring into components
Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:V))
Base.iterate(S::SVD, ::Val{:V}) = (S.V, Val(:done))
Base.iterate(S::SVD, ::Val{:done}) = nothing

function svdvals(A::StaticMatrix)
    sv = svdvals(Matrix(A))
    # We should be using `T2=eltype(sv)`, but it's not inferable for complex
    # eltypes.  See https://github.com/JuliaLang/julia/pull/22443
    T = eltype(A)
    T2 = promote_type(Float32, real(typeof(one(T)/norm(one(T)))))
    similar_type(A, T2, Size(diagsize(A)))(sv)
end

# Implement direct call to BLAS functions that computes the SVD values for `SMatrix` and
# `MMatrix` reducing allocations. In this case, we use `MMatrix` to call the library and
# convert the result back to the input type. Since the former does not exit this scope, we
# can reduce allocations.
#
# We are implementing here the following functions:
#
#   svdvals(A::SMatrix{M, N, Float64}) where {M, N}
#   svdvals(A::SMatrix{M, N, Float32}) where {M, N}
#   svdvals(A::MMatrix{M, N, Float64}) where {M, N}
#   svdvals(A::MMatrix{M, N, Float32}) where {M, N}
#
for (gesdd, elty) in ((:dgesdd_, :Float64), (:sgesdd_, :Float32)),
    (mtype, vtype) in ((SMatrix, SVector), (MMatrix, MVector))

    @eval begin
        function svdvals(A::$mtype{M, N, $elty}) where {M, N}
            K = min(M, N)

            # Convert the input to a `MMatrix` and allocate the required arrays.
            Am    = MMatrix{M, N, $elty}(A)
            Sm    = MVector{K, $elty}(undef)

            # We compute the `lwork` (size of the work array) by obtaining the maximum value
            # from the possibilities shown in:
            #   https://docs.oracle.com/cd/E19422-01/819-3691/dgesdd.html
            lwork = max(8N, 3N + max(M, 7N), 8M, 3M + max(N, 7M))
            work  = MVector{lwork, $elty}(undef)
            iwork = MVector{8min(M, N), BLAS.BlasInt}(undef)
            info  = Ref(1)

            ccall(
                (BLAS.@blasfunc($gesdd), libblastrampoline),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{BLAS.BlasInt},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ptr{C_NULL},
                    Ref{BLAS.BlasInt},
                    Ptr{C_NULL},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ref{BLAS.BlasInt},
                    Ptr{BLAS.BlasInt},
                    Ptr{BLAS.BlasInt},
                    Clong
                ),
                'N',
                M,
                N,
                Am,
                M,
                Sm,
                C_NULL,
                M,
                C_NULL,
                K,
                work,
                lwork,
                iwork,
                info,
                1
            )

            # Check if the return result of the function.
            LAPACK.chklapackerror(info.x)

            # Convert the vector to static arrays and return.
            S = $vtype{K, $elty}(Sm)

            return S
        end
    end
end

# For matrices with interger numbers, we should promote them to float and call `svdvals`.
@inline svdvals(A::StaticMatrix{<: Any, <: Any, <: Integer}) = svdvals(float(A))

# `@inline` annotation is required to propagate `full` as constant to `_svd`
@inline svd(A::StaticMatrix; full=Val(false)) = _svd(A, full)

# Allow plain Bool in addition to Val
# Required inline as of version 1.5 to ensure Bool usage like svd(A,
# full=false) is constant-propagated
@inline _svd(A, full) = _svd(A, Val(convert(Bool, full)))

function _svd(A, full::Val{false})
    f = svd(Matrix(A), full=false)
    U = similar_type(A,  eltype(f.U),  Size(Size(A)[1], diagsize(A)))(f.U)
    S = similar_type(A,  eltype(f.S),  Size(diagsize(A)))(f.S)
    Vt = similar_type(A, eltype(f.Vt), Size(diagsize(A), Size(A)[2]))(f.Vt)
    SVD(U,S,Vt)
end

function _svd(A, full::Val{true})
    f = svd(Matrix(A), full=true)
    U = similar_type(A,  eltype(f.U),  Size(Size(A)[1], Size(A)[1]))(f.U)
    S = similar_type(A,  eltype(f.S),  Size(diagsize(A)))(f.S)
    Vt = similar_type(A, eltype(f.Vt), Size(Size(A)[2], Size(A)[2]))(f.Vt)
    SVD(U,S,Vt)
end

# Implement direct call to BLAS functions that computes the SVD for `SMatrix` and `MMatrix`
# reducing allocations. In this case, we use `MMatrix` to call the library and convert the
# result back to the input type. Since the former does not exit this scope, we can reduce
# allocations.
#
# We are implementing here the following functions:
#
#   _svd(A::SMatrix{M, N, Float64}, full::Val{false}) where {M, N}
#   _svd(A::SMatrix{M, N, Float64}, full::Val{true})  where {M, N}
#   _svd(A::SMatrix{M, N, Float32}, full::Val{false}) where {M, N}
#   _svd(A::SMatrix{M, N, Float32}, full::Val{true})  where {M, N}
#   _svd(A::MMatrix{M, N, Float64}, full::Val{false}) where {M, N}
#   _svd(A::MMatrix{M, N, Float64}, full::Val{true})  where {M, N}
#   _svd(A::MMatrix{M, N, Float32}, full::Val{false}) where {M, N}
#   _svd(A::MMatrix{M, N, Float32}, full::Val{true})  where {M, N}
#
for (gesvd, elty) in ((:dgesvd_, :Float64), (:sgesvd_, :Float32)),
    full in (false, true),
    (mtype, vtype) in ((SMatrix, SVector), (MMatrix, MVector))

    @eval begin
        function _svd(A::$mtype{M, N, $elty}, full::Val{$full}) where {M, N}
            K = min(M, N)

            # Convert the input to a `MMatrix` and allocate the required arrays.
            Am    = MMatrix{M, N, $elty}(A)
            Um    = MMatrix{M, $(full ? :M : :K), $elty}(undef)
            Sm    = MVector{K, $elty}(undef)
            Vtm   = MMatrix{$(full ? :N : :K), N, $elty}(undef)
            lwork = max(3min(M, N) + max(M, N), 5min(M, N))
            work  = MVector{lwork, $elty}(undef)
            info  = Ref(1)

            ccall(
                (BLAS.@blasfunc($gesvd), libblastrampoline),
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BLAS.BlasInt},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ref{BLAS.BlasInt},
                    Ptr{$elty},
                    Ref{BLAS.BlasInt},
                    Ptr{BLAS.BlasInt},
                    Clong,
                    Clong
                ),
                $(full ? 'A' : 'S'),
                $(full ? 'A' : 'S'),
                M,
                N,
                Am,
                M,
                Sm,
                Um,
                M,
                Vtm,
                $(full ? :N : :K),
                work,
                lwork,
                info,
                1,
                1
            )

            # Check if the return result of the function.
            LAPACK.chklapackerror(info.x)

            # Convert the matrices to the correct type and return.
            U  = $mtype{M, $(full ? :M : :K), $elty}(Um)
            S  = $vtype{K, $elty}(Sm)
            Vt = $mtype{$(full ? :N : :K), N, $elty}(Vtm)

            return SVD(U, S, Vt)
        end
    end
end

# For matrices with interger numbers, we should promote them to float and call `svd`.
@inline svd(A::StaticMatrix{<: Any, <: Any, <: Integer}) = svd(float(A))

function \(F::SVD, B::StaticVecOrMat)
    sthresh = eps(F.S[1])
    Sinv = map(s->s < sthresh ? zero(1/sthresh) : 1/s, F.S)
    return transposemult(F.Vt, diagmult(Sinv, transposemult(F.U, B)))
end

transposemult(U, B) = transposemult(Size(U), Size(B), U, B)
function transposemult(sU, sB, U, B)
    sU[1] == sB[1] && return U'*B
    return U[SOneTo(sB[1]),:]'*B
end
diagmult(d, B) = diagmult(Size(d), Size(B), d, B)
function diagmult(sd, sB, d, B)
    sd[1] == sB[1] && return Diagonal(d)*B
    ind = SOneTo(sd[1])
    return isa(B, AbstractVector) ? Diagonal(d)*B[ind] : Diagonal(d)*B[ind,:]
end

