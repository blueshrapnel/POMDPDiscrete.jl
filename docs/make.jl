using POMDPDiscreteGridWorld
using Documenter

DocMeta.setdocmeta!(POMDPDiscreteGridWorld, :DocTestSetup, :(using POMDPDiscreteGridWorld); recursive=true)

makedocs(;
    modules=[POMDPDiscreteGridWorld],
    authors="Karen Archer",
    repo="https://github.com/blueshrapnel/POMDPDiscrete.jl/blob/{commit}{path}#{line}",
    sitename="POMDPDiscrete.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://blueshrapnel.github.io/POMDPDiscrete.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/blueshrapnel/POMDPDiscrete.jl",
    devbranch="main",
)
