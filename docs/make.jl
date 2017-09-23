using Documenter, StaticArrays

makedocs(
         format = :html,
         modules = [StaticArrays],
         sitename = "StaticArrays.jl",
         pages = [
            "Home" => "index.md",
            "API" => "pages/api.md",
            "Quick Start" => "pages/quickstart.md",
            ],
        )

deploydocs(
    repo = "github.com/JuliaArrays/StaticArrays.jl.git",
    julia = "0.6",
    target = "build",
    deps   = nothing,
    make   = nothing
)
