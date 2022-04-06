
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