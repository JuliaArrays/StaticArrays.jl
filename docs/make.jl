using Documenter
using StaticArrays

makedocs(;
    modules = [StaticArrays],
    format = Documenter.HTML(
        canonical = "https://JuliaArrays.github.io/StaticArrays.jl/stable/",
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "pages/api.md",
        "Quick Start" => "pages/quickstart.md",
        ],
    repo = "https://github.com/JuliaArrays/StaticArrays.jl/blob/{commit}{path}#L{line}",
    sitename = "StaticArrays.jl",
)

deploydocs(; repo = "github.com/JuliaArrays/StaticArrays.jl")
