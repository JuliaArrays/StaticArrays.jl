# Maybe this is too general - codegen is hard

immutable StaticIndex{Idx, T} <: StaticVector{T}
    function StaticIndex()
        check_staticindex_params(Idx,T)
        new()
    end
end

(::Type{StaticIndex{Idx}}){Idx}() = StaticIndex{Idx,eltype(Idx)}()

check_staticindex_params(a,b) = error("Element type of $(StaticIndex{a,b}) doesn't match")
check_staticindex_params{T}(::AbstractVector{T},::Type{T}) = nothing
check_staticindex_params{N,T}(::NTuple{N,T},::Type{T}) = nothing

@generated size{Idx}(::StaticIndex{Idx}) = (length(Idx),)
@generated size{Idx}(::Type{StaticIndex{Idx}}) = (length(Idx),)
@generated size{Idx,T}(::Type{StaticIndex{Idx,T}}) = (length(Idx),)

getindex{Idx}(::StaticIndex{Idx}, i) = @inbounds return Idx[i]
@generated getindex{Idx,i}(::StaticIndex{Idx}, ::Type{Val{i}}) = @inbounds return Idx[i]

similar_type{Idx,T}(::Type{StaticIndex{Idx,T}}) = SVector{length(Idx),T}
similar_type{Idx,T1,T2}(::Type{StaticIndex{Idx,T1}},::Type{T2}) = SVector{length(Idx),T2}

# This is a simpler case
immutable StaticOneTo{N} <: StaticVector{Int}
    function StaticIndex()
        check_StaticOneTo_params(N)
        new()
    end
end

check_StaticOneTo_params(N::Int) = nothing
check_StaticOneTo_params(N) = error("StaticOneTo must take an Int")

# Not sure this one is working?
@pure StaticOneTo(N::Int) = StaticOneTo{N}()

@generated size{N}(::StaticOneTo{N}) = (N,)
@generated size{N}(::Type{StaticOneTo{N}}) = (N,)

@pure getindex(::StaticOneTo, i) = i

similar_type{N}(::Type{StaticOneTo{N}}) = SVector{N,Int}
similar_type{N,T}(::Type{StaticOneTo{N}},::Type{T}) = SVector{N,T}
