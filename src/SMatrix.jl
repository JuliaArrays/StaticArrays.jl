immutable SMatrix{S1, S2, T, L} <: StaticMatrix{T}
    data::NTuple{L, T}

    function SMatrix(d::NTuple{L,T})
        check_smatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new(d)
    end

    function SMatrix(d::Tuple)
        check_smatrix_params(Val{S1}, Val{S2}, T, Val{L})
        new(convert_ntuple(T, d))
    end
end

@generated function check_smatrix_params{S1,S2,L}(::Type{Val{S1}}, ::Type{Val{S2}}, T, ::Type{Val{L}})
    if !(T <: DataType) # I think the way types are handled in generated fnctions might have changed in 0.5?
        return :(error("SMatrix: Parameter T must be a DataType. Got $T"))
    end

    if !isa(S1, Int) || !isa(S2, Int) || !isa(L, Int) || S1 < 0 || S2 < 0 || L < 0
        return :(error("SMatrix: Sizes must be positive integers. Got $S1 × $S2 ($L elements)"))
    end

    if S1*S2 == L
        return nothing
    else
        str = "Size mismatch in SMatrix. S1 = $S1, S2 = $S2, but recieved $L elements"
        return :(error(str))
    end
end

@generated function (::Type{SMatrix{S1}}){S1,L}(x::NTuple{L})
    S2 = div(L, S1)
    if S1*S2 != L
        error("Incorrect matrix sizes. $S1 does not divide $L elements")
    end
    T = promote_tuple_eltype(x)

    return quote
        $(Expr(:meta, :inline))
        SMatrix{S1, $S2, $T, L}(x)
    end
end

@generated function (::Type{SMatrix{S1,S2}}){S1,S2,L}(x::NTuple{L})
    T = promote_tuple_eltype(x)

    return quote
        $(Expr(:meta, :inline))
        SMatrix{S1, S2, $T, L}(x)
    end
end

@generated function (::Type{SMatrix{S1,S2,T}}){S1,S2,T,L}(x::NTuple{L})
    return quote
        $(Expr(:meta, :inline))
        SMatrix{S1, S2, T, L}(x)
    end
end

@inline convert{S1,S2,T}(::Type{SMatrix{S1,S2}}, a::AbstractArray{T}) = SMatrix{S1,S2,T}((a...))

#=
@inline (::Type{SMatrix{S1}}){S1}(x1) = SMatrix{S1}((x1,))
@inline (::Type{SMatrix{S1}}){S1}(x1,x2) = SMatrix{S1}((x1,x2))
@inline (::Type{SMatrix{S1}}){S1}(x1,x2,x3) = SMatrix{S1}((x1,x2,x3))
@inline (::Type{SMatrix{S1}}){S1}(x1,x2,x3,x4) = SMatrix{S1}((x1,x2,x3,x4))
@inline (::Type{SMatrix{S1}}){S1}(x...) = SMatrix{S1}(x)

@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1) = SMatrix{S1,S2}((x1,))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1,x2) = SMatrix{S1,S2}((x1,x2))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1,x2,x3) = SMatrix{S1,S2}((x1,x2,x3))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x1,x2,x3,x4) = SMatrix{S1,S2}((x1,x2,x3,x4))
@inline (::Type{SMatrix{S1,S2}}){S1,S2}(x...) = SMatrix{S1,S2}(x)

@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1) = SMatrix{S1,S2,T}((x1,))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1,x2) = SMatrix{S1,S2,T}((x1,x2))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1,x2,x3) = SMatrix{S1,S2,T}((x1,x2,x3))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x1,x2,x3,x4) = SMatrix{S1,S2,T}((x1,x2,x3,x4))
@inline (::Type{SMatrix{S1,S2,T}}){S1,S2,T}(x...) = SMatrix{S1,S2,T}(x)
=#
#####################
## SMatrix methods ##
#####################

@pure size{S1,S2}(::Union{SMatrix{S1,S2},Type{SMatrix{S1,S2}}}) = (S1, S2)
@pure size{S1,S2,T}(::Type{SMatrix{S1,S2,T}}) = (S1, S2)
@pure size{S1,S2,T,L}(::Type{SMatrix{S1,S2,T,L}}) = (S1, S2)

function getindex(v::SMatrix, i::Integer)
    Base.@_inline_meta
    v.data[i]
end

@inline Tuple(v::SMatrix) = v.data
