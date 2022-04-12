module POMDPDiscrete

using POMDPs
using POMDPModelTools

using Plots
using ColorSchemes

export 
    GridWorld,
    State,
    plot_grid_world



include("gridworld.jl")
include("gridworld_visualisation.jl")


end
