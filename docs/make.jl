using Documenter
using StaticArrays
using StaticArraysCore

# Setup for doctests in docstrings
DocMeta.setdocmeta!(StaticArrays, :DocTestSetup, :(using LinearAlgebra, StaticArrays))

makedocs(;
    modules = [StaticArrays, StaticArraysCore],
    format = Documenter.HTML(
        canonical = "https://JuliaArrays.github.io/StaticArrays.jl/stable/",
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "pages/api.md",
        "Quick Start" => "pages/quickstart.md",
        ],
    sitename = "StaticArrays.jl",
)

deploydocs(; repo = "github.com/JuliaArrays/StaticArrays.jl")
