module POMDPDiscrete

using POMDPs
using POMDPModelTools
using POMDPPolicies

using Plots
using ColorSchemes

export
    GridWorld,
    State,
    plot_grid_world,
    random_stochastic_policy,
    uniform_stochastic_policy



include("gridworld.jl")
include("gridworld_visualisation.jl")
include("stochastic_policy.jl")


end
