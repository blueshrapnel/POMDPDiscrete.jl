using POMDPDiscrete
using POMDPs
using POMDPModelTools
using Plots


mdp=GridWorld(
        size=(3,3),
        p_transition = 0.7,
        absorbing_states=[State(1,1), State(3,3)],
        γ = 0.9
    )
V = POMDPDiscrete.value_iteration(mdp)

# creating greedy stochastic policy from optimal value
greedy_policy = POMDPDiscrete.greedy_policy(mdp, V)
# for plotting reshape V
p = render(
    mdp,
    policy=greedy_policy,
    utility=reshape(V, mdp.size),
    title="optimal policy");
savefig(p, "render_sample_optimal_plot.png")
