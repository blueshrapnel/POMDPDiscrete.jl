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
	s::Union{State, AbstractRNG, Nothing}=nothing,
	policy::Union{Policy, Nothing}=nothing,
	title=nothing,
    utility=nothing)

	xmax, ymax = mdp.size
    utility_empty = false
    fillalpha = 0.5
    # plot util as a heatmap
    if isnothing(utility)
        utility_empty=true
        utility = zeros(mdp.size)
        fillalpha=0
    end
    # TODO work out why the heatmap is missing in rect grid
    fig = heatmap(
        utility,
        legend=:none,
        aspect_ratio=:equal,
        framestyle=:box,
        tickdirection=:out,
        fillalpha=fillalpha,
        color=roma.colors)
    xlims!(0.5, xmax+0.5)
    ylims!(0.5, ymax+0.5)
    xticks!(1:xmax)
    yticks!(1:ymax)

    if !utility_empty
        plot!(fillalpha=0.5)
        ann = [(x,y, text(round(utility[x,y], digits=2), 10, :black, :center))
            for x in 1:xmax for y in 1:ymax]
        annotate!(ann)

    end
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

    isa(s, AbstractRNG) ? s = rand(initialstate(mdp)) : nothing
	# plot agent location if a state is specified or a rng provided
	if isnothing(s)
        nothing
    elseif isa(s, State)
	    agent = circle(s.x, s.y, 0.4)
	    plot!(agent, color=:darkgray, fillalpha=0.7)

    end
	# add title to plot
	isnothing(title) ? nothing : title!(title)
	return fig
end

function POMDPModelTools.render(mdp::GridWorld, step=nothing;
    s::Union{State, AbstractRNG, Nothing}=nothing,
    policy::Union{Policy, Nothing}=nothing,
    title=nothing,
    utility=nothing)
    return plot_grid_world(
        mdp::MDP;
        s=s,
        policy=policy,
        title=title,
        utility=utility)
end

"""
    policy_grid(mdp, policy)
    policy_grid(mdp, xmax, ymax)

Return a representation of the policy using unicode arrows stored in a Matrix{String}, where each state is represented by an element of the matrix.  This only shows one action per state, samples distribution to select action.
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
