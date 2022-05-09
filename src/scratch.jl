using POMDPDiscrete
using POMDPs
using POMDPModelTools
using Plots

import POMDPDiscrete.InformationChannel
import POMDPDiscrete.relevant_information_policy


mdp=GridWorld(
        size=(3,3),
        p_transition = 0.7,
        absorbing_states=[State(1,1), State(3,3)],
        γ = 0.9
    )
V = POMDPDiscrete.value_iteration(mdp)

# creating greedy stochastic policy from optimal value
greedy_policy = POMDPDiscrete.greedy_policy(mdp, V)

channel = InformationChannel(mdp)

RI_policy, Z = POMDPDiscrete.relevant_information_policy(channel, mdp)

# for plotting reshape V
p = render(
    mdp,
    policy=RI_policy,
    utility=reshape(V, mdp.size),
    title="relevant information policy");
savefig(p, "render_sample_RI_plot.png")
