@inline det(A::StaticMatrix) = _det(Size(A), A)
@inline logdet(A::StaticMatrix) = _logdet(Size(A), A)

@inline _det(::Size{(1,1)}, A::StaticMatrix) = @inbounds return A[1]

@inline function _det(::Size{(2,2)}, A::StaticMatrix)
    @inbounds return A[1]*A[4] - A[3]*A[2]
end

@inline function _det(::Size{(3,3)}, A::StaticMatrix)
    @inbounds x0 = SVector(A[1], A[2], A[3])
    @inbounds x1 = SVector(A[4], A[5], A[6])
    @inbounds x2 = SVector(A[7], A[8], A[9])
    return vecdot(x0, cross(x1, x2))
end

@inline function _det(::Size{(2,2)}, A::StaticMatrix{2, 2, <:Unsigned})
    @inbounds return det(SMatrix{2,2,Int64}(A))
end

@inline function _det(::Size{(3,3)}, A::StaticMatrix{3, 3, <:Unsigned})
    @inbounds return det(SMatrix{3,3,Int64}(A))
end

@inline function _det(::Size{(4,4)}, A::StaticMatrix{4, 4, <:Unsigned})
    @inbounds return det(SMatrix{4,4,Int64}(A))
end

@inline _logdet(S::Union{Size{(1,1)},Size{(2,2)},Size{(3,3)},Size{(4,4)}}, A::StaticMatrix) = log(_det(S, A))

for (symb, f) in [(:_det, :det), (:_logdet, :logdet)]
    eval(quote
        @generated function $symb{S}(::Size{S}, A::StaticMatrix)
            if S[1] != S[2]
                throw(DimensionMismatch("matrix is not square"))
            end
            return quote # Implementation from Base
                @_inline_meta
                T = eltype(A)
                T2 = typeof((one(T)*zero(T) + zero(T))/one(T))
                if istriu(A) || istril(A)
                    return convert(T2, $($f)(UpperTriangular(A))) # Is this a Julia bug that a convert is not type stable??
                end
                AA = convert(Array{T2}, A)
                return $($f)(lufact(AA))
            end
        end
    end)
end
