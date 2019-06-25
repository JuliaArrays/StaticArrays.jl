using Documenter, StaticArrays

makedocs(
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
)
