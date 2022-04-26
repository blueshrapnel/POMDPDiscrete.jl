using POMDPDiscrete
using POMDPs
using POMDPModelTools

using Random

mdp = GridWorld(
    size=(5,7),
    absorbing_states=[State(1,1)],
    p_transition = 1.0,
    γ = 1.0)

Vopt = POMDPDiscrete.value_iteration(mdp)
policy = POMDPDiscrete.greedy_policy(mdp, Vopt)

# test plot of the gridworld and agent location
render(mdp, s=State(1,1), utility=Vopt, policy=policy)

