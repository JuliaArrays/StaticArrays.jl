module StaticArraysChainRulesCoreExt

using StaticArrays
# ChainRulesCore imports
import ChainRulesCore: ProjectTo, Tangent, project_type, rrule
import ChainRulesCore as CRC

# Projecting a tuple to SMatrix leads to ChainRulesCore._projection_mismatch by default, so
# overloaded here
function (project::ProjectTo{<:Tangent{<:Tuple}})(dx::StaticArraysCore.SArray)
    dy = reshape(dx, axes(project.elements))
    dz = ntuple(i -> project.elements[i](dy[i]), length(project.elements))
    return project_type(project)(dz...)
end

# Project SArray to SArray
function ProjectTo(x::SArray{S, T}) where {S, T}
    return ProjectTo{SArray}(; element = CRC._eltype_projectto(T), axes = S)
end

function (project::ProjectTo{SArray})(dx::AbstractArray{S, M}) where {S, M}
    return SArray{project.axes}(dx)
end

# Adjoint for SArray constructor
function rrule(::Type{T}, x::Tuple) where {T <: SArray}
    project_x = ProjectTo(x)
    ∇Array(∂y) = (NoTangent(), project_x(∂y))
    return T(x), ∇Array
end

end
