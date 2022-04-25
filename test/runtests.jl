using SafeTestsets

@safetestset "gridworld state space tests" begin
    include("test_gridworld_vis.jl")
    include("test_gridworld_dynamic_prog.jl")
end

@safetestset "gridworld mdp tests" begin
    include("test_build_probabilistic_model.jl")
    include("test_gridworld_stochastic_policy.jl")
    include("test_gridworld_dynamics.jl")
end
