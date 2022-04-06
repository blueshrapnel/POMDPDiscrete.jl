using POMDPDiscrete
using Documenter

DocMeta.setdocmeta!(POMDPDiscrete, :DocTestSetup, :(using POMDPDiscrete); recursive=true)

makedocs(;
    modules=[POMDPDiscrete],
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
