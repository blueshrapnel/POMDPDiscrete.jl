
#=
MDP Container
=============
holds: 
* 	all the information required to define the MDP tuple (𝒮, 𝒜, T, R, γ)
* 	gridworld parameters, either a custom struct or individual fields
*	some helper fields
* 	𝒮::Vector{Int}          	 # state space
	𝒜::Vector{Int}       	 	 # action space
	T::Array{Float64, 3}   		# transition function, i.e. probabilistic model Pr [s′, a, s]
	R::Matrix{Float64}      	# rewards
=#



#=
Gridworld states and state space
================================
There are several options to deal with terminal states
* 	include is_done as a field in the struct, then you need to check whether s′ is_terminal
create a vector of all combinations of x, y, done
* 	[possibly more flexible] include the is_done in the transition function, then indicate termination
through transitioning to a null_state outside the grid, e.g. -State(-1, -1)
* need to include the null state as part of the state space
* also need a way to identify an absorbing state, e.g. use reward = 0, or a vector of States
=#

struct State
	x::Int64  					# x position
	y::Int64					# y position
end

    # helper functions for working with grid world states
# Base comparison operator to check whether the coordinates of the two states are equal
Base.:(==)(s1::State, s2::State) = (s1.x == s2.x) && (s1.y == s2.y)

# State arithmetic for effecting transitions : Base add operator to add two states 
Base.:(+)(s1::State, s2::State) = State(s1.x + s2.x, s1.y + s2.y)

mutable struct GridWorld <: MDP{State, Symbol}  # MDP{state_type, action_type}
	# parameters
	size::Tuple{Int, Int}      	# size of the grid 
	p_transition::Real         	# probability of successful transition to target
	γ::Real              		# discount factor
	
	absorbing_states::Vector{State} # vector of states which are absorbing
	# for multiple reward_values vector of corresponding values for a list of reward_states
	
	# MDP tuple
	𝒮::Vector{State}			 # state space
	𝒜::Vector{Symbol} 			 # action space
	
	# helper fields 
	ci::CartesianIndices	    # to access stateindex (xy2s)
	next_states::Dict{Tuple{State, Symbol}, Vector{Symbol}}
end

# Default Contstructor for GridWorld mdp container
function GridWorld(
	# parameters
	;size::Tuple{Int, Int}=(5,7), 
	p_transition::Real=0.7,
	absorbing_states::Vector{State}=[State(1,1)],
	γ::Real=1.0)

	# MDP tuple 
	𝒮 = [[State(x,y) for x=1:size[1], y=1:size[2]]...] 
	𝒜 = [:up, :right, :down, :left]
	
	# helpers
	ci = CartesianIndices((size[1], size[2]))
	next_states = Dict()
	return GridWorld(size, p_transition, γ, absorbing_states, 𝒮, 𝒜, ci, next_states)
end

# check whether a state is within the gridworld - move first, questions later
inbounds(mdp::GridWorld, s::State) = 1 ≤ s.x ≤ mdp.size[1] && 1 ≤ s.y ≤ mdp.size[2]
