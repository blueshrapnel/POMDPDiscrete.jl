using POMDPDiscrete

import POMDPDiscrete.one_step_lookahead
import POMDPDiscrete.one_step_lookahead_value
import POMDPDiscrete.one_step_lookahead_actions
import POMDPDiscrete.one_step_lookahead_value_actions

using Test

@testset "test one_step_lookahead" begin
    # deterministic dynamics in a 2x2 grid world
    mdp = GridWorld(
        size=(2,2), 
        absorbing_states = [State(1,1)],
        p_transition=1.0,
        γ=1.0)
    Nₐ = mdp.Nₐ
    Nₛ = mdp.Nₛ
    P = POMDPDiscrete.build_probabilistic_model(mdp)
    R = POMDPDiscrete.get_rewards(mdp)            # as a column vector
    R_s_s′= repeat(R, 1, Nₛ)        # R[s, s'] 

    V = [0, -1, -1, -2]
    # case where there is one best action (si = 2 or 3)
    si = 3                          # top left corner
    qₛ = one_step_lookahead(si, P, R_s_s′ ,mdp.γ, V, Nₐ)
    @test qₛ ≈ [-2, -3, -1, -2]
    best_value = one_step_lookahead_value(si, P, R_s_s′ ,mdp.γ, V, Nₐ)
    best_actions = one_step_lookahead_actions(qₛ)
    @test best_value ≈ -1
    @test best_actions ≈ [3]

    
    best_value, best_actions = one_step_lookahead_value_actions(si, P, R_s_s′ ,mdp.γ, V, Nₐ)
    @test best_value ≈ -1
    @test best_actions ≈ [3]

    # case where there are two best actions (si = 4)
    si = 4
    qₛ = one_step_lookahead(si, P, R_s_s′ ,mdp.γ, V, Nₐ)
    @test qₛ ≈ [-3, -3, -2, -2]
    best_value = one_step_lookahead_value(si, P, R_s_s′ ,mdp.γ, V, Nₐ)
    @test best_value ≈ -2
    best_value, best_actions = one_step_lookahead_value_actions(si, P, R_s_s′ ,mdp.γ, V, Nₐ)
    @test best_value ≈ -2
    @test best_actions ≈ [3, 4]

end