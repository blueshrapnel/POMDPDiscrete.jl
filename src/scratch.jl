using POMDPDiscrete
using POMDPs
using POMDPModelTools
using Plots

import POMDPDiscrete.InformationChannel
import POMDPDiscrete.relevant_information_policy


mdp=GridWorld(
        size=(5,5),
        p_transition = 0.7,
        absorbing_states=[State(1,1), State(5,5)],
        γ = 0.9
    )
V = POMDPDiscrete.value_iteration(mdp)

# creating greedy stochastic policy from optimal value
greedy_policy = POMDPDiscrete.greedy_policy(mdp, V)

channel = InformationChannel(mdp)

β=6.30957
RI_policy, Z = POMDPDiscrete.relevant_information_policy(channel, mdp, β=β; max_iters=25)

# for plotting reshape V
p = render(
    mdp,
    policy=RI_policy,
    utility=reshape(V, mdp.size),
    title="β: $β RI policy");
savefig(p, "sample_RI_policy_plot.png")

display(RI_policy)
