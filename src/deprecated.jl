# Deprecations, included in v0.12, to be removed for v0.13
@deprecate +(a::Number, b::StaticArray) a .+ b
@deprecate +(a::StaticArray, b::Number) a .+ b
@deprecate -(a::Number, b::StaticArray) a .- b
@deprecate -(a::StaticArray, b::Number) a .- b

@deprecate +(a::Number, b::SHermitianCompact) a .+ b
@deprecate +(a::SHermitianCompact, b::Number) a .+ b
@deprecate -(a::Number, b::SHermitianCompact) a .- b
@deprecate -(a::SHermitianCompact, b::Number) a .- b
