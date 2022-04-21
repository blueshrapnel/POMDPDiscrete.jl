"""
    build_probabilistic_model(mdp)

Return a 3 dimensional matrix representing P[s′, a, s], taking into account successor state distributions for action/state pairs.
"""

function build_probabilistic_model(mdp::Union{MDP, POMDP})
    Nₐ = length(actions(mdp))
    Nₛ = length(states(mdp))
    P = zeros(Nₛ, Nₐ, Nₛ)
    sa_pairs =  [repeat(states(mdp), inner=[Nₐ]) repeat(actions(mdp), outer=[Nₛ])]

    for sa in eachrow(sa_pairs)
        s = sa[1]; a = sa[2];
        # i in the variable name denotes index
        si = stateindex(mdp, s)
        ai = actionindex(mdp, a)
        ps′ = transition(mdp, s, a)
        for (s′, p) in weighted_iterator(ps′)
            if p > 0.0
                s′i = stateindex(mdp, s′)
                P[s′i, ai, si] += p
            end
        end
    end
    return P
end
