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
