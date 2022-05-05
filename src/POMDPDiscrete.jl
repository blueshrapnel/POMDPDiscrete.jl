module POMDPDiscrete

using POMDPs
using POMDPModelTools
using POMDPPolicies

using Plots
using ColorSchemes

using Random

using ProgressBars

export
    GridWorld,
    State,
    plot_grid_world,
    random_stochastic_policy,
    uniform_stochastic_policy,
    policy_transition_matrix


include("parameters.jl")
include("gridworld.jl")
include("gridworld_visualisation.jl")
include("stochastic_policy.jl")
include("probabilistic_model.jl")
include("dynamic_programming.jl")


end
