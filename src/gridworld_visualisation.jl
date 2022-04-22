import ColorSchemes.roma

# define a function that returns a cicle as a Plots.Shape
function circle(x, y, r=.2; n=30)
    θ = 0:360÷n:360
    Plots.Shape(r*sind.(θ) .+ x, r*cosd.(θ) .+ y)
end

# define a function that returns a rectangle as a Plots.Shape
# for plotting cells of the grid
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])


function plot_grid_world(
	mdp::MDP;
	s::Union{State, Nothing}=nothing,
	policy::Union{Policy, Nothing}=nothing,
	title::Union{String, Nothing}=nothing)
	xmax, ymax = mdp.size
	empty = zeros(mdp.size)
	fig = heatmap(
		empty,
		legend=:none,
		aspect_ratio=:equal,
		framestyle=:box,
		tickdirection=:out,
		fillalpha=0, color=roma.colors)
	xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

	# plot cell outlines
    for x in 1:xmax, y in 1:ymax
        # display policy on the plot as arrows
		rect = rectangle(1, 1, x - 0.5, y - 0.5)
		plot!(rect, fillalpha=0, linecolor=:gray)
    end

	# plot policy arrows
	if isnothing(policy)
		nothing
	elseif isa(policy, StochasticPolicy)
		GR.setarrowsize(0.5)
		# column major
		xs = repeat(1:xmax, 1, ymax) |> vec
		ys = repeat(1:ymax, 1, xmax)' |> vec

		for a in ordered_actions(mdp)
			action_probs = policy.π[:,actionindex(mdp, a)]
			movement = MOVEMENTS[a]
			us, vs = action_probs .* movement.x, action_probs .* movement.y
			quiver!(xs, ys, quiver=(us, vs), color=:blue)
		end
	end

	# plot agent location
	isnothing(s) ? s = rand(initialstate(mdp)) : nothing
	agent = circle(s.x, s.y, 0.4)
	plot!(agent, color=:darkgray, fillalpha=0.7)

	# add title to plot
	isnothing(title) ? nothing : title!(title)
	return fig
end

function POMDPModelTools.render(mdp::GridWorld, step=nothing;
    s::Union{State, Nothing}=nothing,
    policy::Union{Policy, Nothing}=nothing,
    title::Union{String, Nothing}=nothing)
    return plot_grid_world(
        mdp::MDP;
        s=s,
        policy=policy,
        title=title)
end

"""
    policy_grid(mdp, policy)
    policy_grid(mdp, xmax, ymax)

Return a representation of the policy using unicode arrows stored in a Matrix{String}, where each state is represented by an element of the matrix.
"""
function policy_grid(mdp::GridWorld, policy::Policy)
    xmax, ymax = mdp.size
    return policy_grid(policy, xmax, ymax)
end

function policy_grid(policy::Policy, xmax::Int, ymax::Int)
    arrows = Dict( :up => "↑",
                   :right => "→",
                   :down => "↓",
                   :left => "←")
    grid = Array{String}(undef, xmax, ymax)
    for x in 1:xmax, y in 1:ymax
        s = State(x,y)
		# selects the action from the policy given the state
        grid[x,y] = arrows[action(policy, s)]
    end

    return grid
end
