using Documenter
using StaticArrays
using StaticArraysCore

# Setup for doctests in docstrings
doctest_setup = :(using LinearAlgebra, StaticArrays)
DocMeta.setdocmeta!(StaticArrays,     :DocTestSetup, doctest_setup)
DocMeta.setdocmeta!(StaticArraysCore, :DocTestSetup, doctest_setup)

makedocs(;
    modules = [StaticArrays, StaticArraysCore],
    format = Documenter.HTML(
        canonical = "https://JuliaArrays.github.io/StaticArrays.jl/stable/",
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Quick Start" => "quickstart.md",
        ],
    sitename = "StaticArrays.jl",
)

deploydocs(; repo = "github.com/JuliaArrays/StaticArrays.jl")
