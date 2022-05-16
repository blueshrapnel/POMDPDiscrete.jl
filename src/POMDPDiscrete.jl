module POMDPDiscrete

using POMDPs
using POMDPModelTools
using POMDPPolicies

using Plots
using ColorSchemes

using Random
using LinearAlgebra

export
    GridWorld,
    State,
    plot_grid_world,
    random_stochastic_policy,
    uniform_stochastic_policy,
    policy_transition_matrix,
    relevant_information_policy, 
    InformationChannel


include("parameters.jl")
include("gridworld.jl")
include("gridworld_visualisation.jl")
include("stochastic_policy.jl")
include("probabilistic_model.jl")
include("dynamic_programming.jl")
include("information_utils.jl")
include("information_channel.jl")
include("relevant_information.jl")


end
