
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

# the length of a GridWorld is the number of states
# note this currently assumes every cell in the grid is a state
Base.length(mdp::GridWorld) = prod(mdp.size)

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

#= State space
==============
𝒮 = [[State(x,y) for x=1:mdp.size[1], y=1:mdp.size[2]]..., mdp.null_state]
=#

POMDPs.states(mdp::GridWorld) = mdp.𝒮

POMDPs.stateindex(mdp::GridWorld, s::State) = LinearIndices(mdp.ci)[s.x, s.y]

POMDPs.initialstate(mdp::GridWorld) = Uniform(mdp.𝒮)# Deterministic(State(4,4))

#= Action space
===============
cardinal actions
* 	use `@enum` to represent the actions, this simplifies actionindex, 
	remember though that enum is base 0, so SKIP 0 to avoid BoundsError
	e.g. @enum Action SKIP UP RIGHT DOWN LEFT # synonymous with North, East, South, West - enum is 0 based
* 	so better to use Symbol and then an explicit actionindex function.
*	define a constant movements dictionary to effect action transitions 
=#

POMDPs.actions(mdp::GridWorld) = mdp.𝒜

function POMDPs.actionindex(mdp::GridWorld, a::Union{Symbol}) 
	if a == :up
		return 1
	elseif a == :right
		return 2
	elseif a == :down
		return 3
	elseif a == :left
		return 4
	end
	error("Invalid action in GridWorld: $a")
end

const MOVEMENTS = Dict(
	:up    => State(0, 1), 
	:right => State(1, 0),
	:down  => State(0, -1),
	:left  => State(-1, 0)
);	

#= Reward function
==================
In this case the reward function is dependent only on the state, although typically it is also dependent on the action taken, and potentially also the successor state R(s, a, s′)
*	one possibility would be to store the reward_states as a list of states in which the agent receives a reward, and a corresponding reward_values vector which contains the values of rewards received in those sttes, you could define this in the GridWorld struct
*	another alternative would be to declare a list of absorbing states, these states have a reward of 0 and other states then have a reward of -1
=#
function POMDPs.reward(mdp::GridWorld, s::State, a::Any=nothing)  
	# currently no check for whether the state is in 𝒮 
	# so any state valid or not incurs a cost of -1, this is important for bumping off walls, etc. 
	# define a simple corner goal
	if s ∈ mdp.absorbing_states
		return 0
	else
		return -1
	end
end

POMDPs.isterminal(mdp::GridWorld, s::State) = s ∈ mdp.absorbing_states 

