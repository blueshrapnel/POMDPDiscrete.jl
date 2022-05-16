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

@safetestset "relevant_information tests" begin
    include("test_relevant_information.jl")
end

@safetestset "information utilities" begin
    include("test_information_utils.jl")
    #include("test_channel_capacity.jl")
end
