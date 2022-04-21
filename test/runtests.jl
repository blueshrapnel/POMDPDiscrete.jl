using SafeTestsets

@safetestset "gridworld tests" begin
    include("test_gridworld_dynamics.jl")
    include("test_gridworld_vis.jl")
    include("test_gridworld_stochastic_policy.jl")
    include("test_build_probabilistic_model.jl")
end
