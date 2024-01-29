module StaticArraysChainRulesCoreExt

using StaticArrays
# ChainRulesCore imports
import ChainRulesCore: NoTangent, ProjectTo, Tangent, project_type, rrule
import ChainRulesCore as CRC

# Projecting a tuple to SMatrix leads to CRC._projection_mismatch by default, so
# overloaded here
function (project::ProjectTo{<:Tangent{<:Tuple}})(dx::StaticArraysCore.SArray)
    dy = reshape(dx, axes(project.elements))
    dz = ntuple(i -> project.elements[i](dy[i]), length(project.elements))
    return project_type(project)(dz...)
end

# Project SArray to SArray
function ProjectTo(x::SArray{S, T}) where {S, T}
    # We have a axes field because it is expected by other ProjectTo's like the one for Transpose
    return ProjectTo{SArray}(; element = CRC._eltype_projectto(T), axes = axes(x),
        size = Size(x))
end

@inline _sarray_from_array(::Size{T}, dx::AbstractArray) where {T} = SArray{Tuple{T...}}(dx)

(project::ProjectTo{SArray})(dx::AbstractArray) = _sarray_from_array(project.size, dx)

# Adjoint for SArray constructor
function rrule(::Type{T}, x::Tuple) where {T <: SArray}
    project_x = ProjectTo(x)
    ∇Array(∂y) = (NoTangent(), project_x(∂y))
    return T(x), ∇Array
end

function rrule(::Type{T}, xs::Number...) where {T <: SVector}
    project_x = ProjectTo(xs)
    ∇Array(∂y) = (NoTangent(), project_x(∂y)...)
    return T(xs...), ∇Array
end

end
