#####
##### constructors
#####

ChainRulesCore.@non_differentiable (::Type{T} where {T<:Union{SArray, SizedArray}})(::UndefInitializer, args...)

function ChainRulesCore.frule((_, ẋ), ::Type{T}, x::Tuple) where {T<:Union{SArray, SizedArray}}
    return T(x), T(ẋ)
end

function ChainRulesCore.rrule(::Type{T}, x::Tuple) where {T<:Union{SArray, SizedArray}}
    project_x = ProjectTo(x)
    Array_pullback(ȳ) = (NoTangent(), project_x(ȳ))
    return T(x), Array_pullback
end

function (project::ChainRulesCore.ProjectTo{AbstractArray})(dx::AbstractArray{S,M}) where {S,M}
    # First deal with shape. The rule is that we reshape to add or remove trivial dimensions
    # like dx = ones(4,1), where x = ones(4), but throw an error on dx = ones(1,4) etc.
    dy = if axes(dx) === project.axes
        dx
    else
        for d in 1:max(M, length(project.axes))
            if size(dx, d) != length(get(project.axes, d, 1))
                throw(_projection_mismatch(project.axes, size(dx)))
            end
        end
        reshape(dx, project.axes)
    end
    # Then deal with the elements. One projector if AbstractArray{<:Number},
    # or one per element for arrays of anything else, including arrays of arrays:
    dz = if hasproperty(project, :element)
        T = project_type(project.element)
        S <: T ? dy : map(project.element, dy)
    else
        map((f, y) -> f(y), project.elements, dy)
    end
    return dz
end