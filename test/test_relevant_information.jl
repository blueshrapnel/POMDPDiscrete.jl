using POMDPDiscrete

import POMDPDiscrete.InformationChannel

using Test

@testset "small deterministic grid world" begin

    mdp=GridWorld(
        size=(5, 5),
        p_transition = 1,
        absorbing_states=[State(3,3)],
        γ = 1.0
    )

    channel = InformationChannel(mdp)

    β = 100
    RI_policy, Q, Z = POMDPDiscrete.relevant_information_policy(channel, mdp, β=β; max_iters=25)

    @test sum(RI_policy) ≈ length(mdp.𝒮)

    # checking against values optained using python GridFour code
    display(RI_policy)

    display(Q)
    # results of Q calculated on a similar grid in Python code
    # up and down actions switched
    gridFour_Q =([
                [-4 -4 -5 -5];
                [-3 -3 -4 -5];
                [-2 -4 -3 -4];
                [-3 -5 -4 -3];
                [-4 -5 -5 -4];
                [-3 -3 -5 -4];
                [-2 -2 -4 -4];
                [-1 -3 -3 -3];
                [-2 -4 -4 -2];
                [-3 -4 -5 -3];
                [-4 -2 -4 -3];
                [-3 -1 -3 -3];
                [ 0  0  0  0];
                [-3 -3 -3 -1];
                [-4 -3 -4 -2];
                [-5 -3 -3 -4];
                [-4 -2 -2 -4];
                [-3 -3 -1 -3];
                [-4 -4 -2 -2];
                [-5 -4 -3 -3];
                [-5 -4 -4 -5];
                [-4 -3 -3 -5];
                [-3 -4 -2 -4];
                [-4 -5 -3 -3];
                [-5 -5 -4 -4]])
    @test round.(Int, Q) == gridFour_Q

end
