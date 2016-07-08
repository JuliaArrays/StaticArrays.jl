import Base: *

# TODO size-inferrable products with AbstractArray (such as StaticMatrix * AbstractVector)
# TODO A*_mul_B*

@generated function *(A::StaticMatrix, b::StaticVector)
    sA = size(A)
    sb = size(b)
    TA = eltype(A)
    Tb = eltype(b)

    s = (sA[1],)
    T = promote_type(TA, Tb)

    if sb[1] != sA[2]
        error("Dimension mismatch")
    end

    if s == sb
        if T == Tb
            newtype = b
        else
            newtype = similar_type(b, T)
        end
    else
        if T == Tb
            newtype = similar_type(b, s)
        else
            newtype = similar_type(b, T, s)
        end
    end

    exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(A[$k, $j]*b[$j]) for j = 1:sA[2]]) for k = 1:sA[1]]

    return quote
        $(Expr(:meta,:inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end

@generated function *(A::StaticMatrix, B::StaticMatrix)
    sA = size(A)
    sB = size(B)
    TA = eltype(A)
    TB = eltype(B)

    s = (sA[1], sB[2])
    T = promote_type(TA, TB)

    if sB[1] != sA[2]
        error("Dimension mismatch")
    end

    # TODO think about which to be similar to
    if s == sB
        if T == TB
            newtype = B
        else
            newtype = similar_type(B, T)
        end
    else
        if T == TB
            newtype = similar_type(B, s)
        else
            newtype = similar_type(B, T, s)
        end
    end

    exprs = [reduce((ex1,ex2) -> :(+($ex1,$ex2)), [:(A[$k1, $j]*B[$j,$k2]) for j = 1:sA[2]]) for k1 = 1:sA[1], k2 = 1:sB[2]]

    return quote
        $(Expr(:meta,:inline))
        @inbounds return $(Expr(:call, newtype, Expr(:tuple, exprs...)))
    end
end
