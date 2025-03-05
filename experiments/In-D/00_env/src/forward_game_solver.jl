ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
ENV["JULIA_PKG_DISABLE_EXTENSIONS"] = "1"
ENV["JULIA_PKG_QUIET"] = "1"

using CSV
using FileIO
using ImageCore 
using CairoMakie
using ParametricMCPs
using Symbolics: scalarize, @variables, Symbolics
using SparseArrays: findnz
using TrajectoryGamesBase
using TrajectoryGamesExamples
using BlockArrays
using LinearAlgebra
using Statistics: mean
using ChainRulesCore: ChainRulesCore
using ForwardDiff: ForwardDiff
using Zygote: Zygote
using Infiltrator
include("./Env00.jl")
include("./ProblemFormulation.jl")
include("./Solve.jl")

"""
Main function to run the forward game solver with default parameters.
"""
function main()
    # Run with default parameters
    forward_solution, init, fig = run_forward_game(
        # data_id = "00",
        # tracks = [36], # left
        # tracks = [37], # bottom
        # tracks = [57], # right
        # downsample_rate = 14,
        verbose = true, 
        show_plot = true
    )
    
    println("Forward game solved successfully!")
    # return forward_solution, init, fig
end

"""
    run_forward_game(;
        data_id = "00",
        tracks = [36],
        downsample_rate = 14,
        full_state = true, 
        verbose = true, 
        show_plot = true
    )

Pull data from the data directory, instantiate a forward game with bicycle dynamics,
collision constraints, goal-finding costs, and the provided environment equations,
solve the forward game, and plot the trajectories on the environment.

Parameters:
- `data_id`: ID of the InD dataset to use (e.g., "07")
- `tracks`: Array of track IDs to include in the game
- `downsample_rate`: Rate at which to downsample the trajectory data
- `full_state`: Whether to use full state information
- `verbose`: Whether to print detailed information
- `show_plot`: Whether to display the plot

Returns a tuple containing:
- `forward_solution`: The solution to the forward game
- `init`: The game initialization parameters
- `fig`: The figure object with plotted trajectories
"""
function run_forward_game(;
    data_id = "00",
    tracks = [36],
    downsample_rate = 10,
    verbose = true, 
    show_plot = true
)
    # # 1. Pull data from the data directory
    # observations = pull_trajectory(data_id;
    #     track = tracks, 
    #     downsample_rate = downsample_rate, 
    #     all = false
    # )
    # 2. Set up bicycle dynamics
    dynamics = BicycleDynamics(;
        dt = 0.04 * downsample_rate, # Time step based on framerate and downsample rate
        l = 1.0, # Vehicle length
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]), # Speed and steering angle bounds
        integration_scheme = :forward_euler
    )
    # 3. Initialize the game
    init = init_game(;
        # initial_state = observations[1],
        initial_state = [700, 900, 10, 3*pi/2],
        # goals = [observations[end][Block(i)][1:2] for i in 1:length(tracks)],
        goals = [[500, 450]], # left
        # Use our new environment function that uses equations from env.jl
        # horizon=length(observations),
        horizon = 20,
        dt = 0.04 * downsample_rate,
        verbose = verbose,
        dynamics = dynamics,
    )
    
    # Print information about the game setup if verbose
    if verbose
        println("Initial state: ", init.initial_state)
        println("Player Goals: ", init.goals)
        println("Horizon: ", init.horizon)
        println("Game initialized, initializing MCP coupled optimization solver")
    end
    
    # 4. Initialize the solver
    mcp_solver = MCPCoupledOptimizationSolver(
        init.game_structure,
        init.horizon,
        blocksizes(init.goals, 1)
    )
    
    if verbose
        println("MCP coupled optimization solver initialized")
        println("Solving forward game...")
    end
    
    # 5. Solve the forward game
    fs_temp = solve_mcp_game(
        mcp_solver.mcp_game,
        init.initial_state,
        init.goals;
        verbose = false
    )

    verbose && println("Forward game solved, status: ", fs_temp.status)
    
    # 6. Reconstruct the solution
    forward_solution = reconstruct_solution(
        fs_temp,
        init.game_structure,
        init.horizon
    )
    
    # 7. Plot the trajectories on the environment
    fig = nothing
    if show_plot
        # Create figure with environment and trajectories
        fig = plot_trajectories(
            observations, 
            forward_solution, 
            tracks, 
            init.horizon;
            data_id = data_id
        )
    end
    
    return forward_solution, init, fig
end

"""
    plot_trajectories(observations, forward_observations, tracks, horizon; data_id = "07")

Plot the observed and forward solution trajectories on the environment.

Parameters:
- `observations`: Array of observations
- `forward_observations`: Array of forward solution observations
- `tracks`: Array of track IDs
- `horizon`: Number of time steps
- `data_id`: ID of the InD dataset

Returns the figure object with plotted trajectories.
"""
function plot_trajectories(observations, forward_observations, tracks, horizon; data_id = "07")
    # Create figure with road boundaries using the plotting functions from env.jl
    # First generate the road equations
    equations = Env00.generate_road_equations(circles = true, ellipses = false, lines = true)
    
    # Use the plot_background_with_equations function from env.jl to create the base figure
    fig, ax = Env00.plot_background_with_equations(
        equations = equations, 
        resolution = (1000, 1000),
        show_plot = false,  # Don't display yet
        remove_black = true
    )
    
    # Plot the trajectories
    colors = [(:red, 1.0), (:blue, 1.0), (:green, 1.0)]
    
    # Plot observed trajectories
    for i in eachindex(tracks)
        lines!(ax, 
            [observations[t][Block(i)][1] for t in 1:horizon],
            [observations[t][Block(i)][2] for t in 1:horizon], 
            color = colors[i], 
            linewidth = 2,
            label = "Observed Track $(tracks[i])")
    end
    
    # Plot forward solution trajectories
    for i in eachindex(tracks)
        lines!(ax, 
            [forward_observations[t][Block(i)][1] for t in 1:horizon],
            [forward_observations[t][Block(i)][2] for t in 1:horizon], 
            color = (colors[i][1], 0.5), 
            linewidth = 2,
            linestyle = :dash,
            label = "Forward Solution Track $(tracks[i])")
    end
    
    # Add legend
    axislegend(ax, position = :lt)
    
    # Add title
    ax.title = "Forward Game Solution vs. Observed Trajectories"
    
    # Display the figure
    display(fig)
    
    # Save the figure
    save("forward_game_solution.png", fig)
    
    return fig
end
function norm_sqr(x)
    x' * x
end
function basic_game_structure(
    num_players,
    goals;
    dynamics = BicycleDynamics(;
        dt = 0.04, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/2], ub = [5, pi/2]),
        integration_scheme = :forward_euler
    ),
    environment = get_env(),
)
    cost = let
        function target_cost(x, i)
            norm_sqr(x[1:2] - goals[i])
        end
        function control_cost(u)
            norm_sqr(u)
        end
        # function lane_center_cost(x, i, context_state, t)
        #     (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[2] - lane_centers[i](x[1]))
        # end
        function cost_for_player(i, xs, us, context_state, T)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], i) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)]) for t in 1:T])

            1.0 * mean_target +
            0.1 * control
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            T = size(us,1)
            [cost_for_player(i, xs, us, context_state, T) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    # dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    TrajectoryGame(
        dynamics,
        cost,
        environment,
        nothing,
    )
end

function init_game(;
    state_dim = 4,
    action_dim = 2,
    game_environment = Env00.get_env(), # Use our environment function with equations from env.jl
    initial_state = mortar([
        [700, 900, 10, 3*pi/2],
    ]),
    goals = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ],
    dt = 0.04,
    verbose = false,
    dynamics = BicycleDynamics(;
        dt = dt, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/2], ub = [5, pi/2]),
        integration_scheme = :forward_euler
    ),
    lane_centers = nothing,
    horizon = 10
)
    !verbose || print("initializing game ... ")
    game_structure = basic_game_structure(
        1, #number of players
        goals; # goals
        environment = game_environment, # Use the provided environment directly
        dynamics = dynamics,
    )
    !verbose || print(" game structure initialized\n")
    observation_dim = state_dim
    observation_model = 
        (x; σ = σ_) -> vcat([x[state_dim * (i - 1)+1:state_dim*i] .+ σ * randn(state_dim) for i in 1:1]...)
    !verbose || println("observation model initialized")

    (;
    initial_state = initial_state,
    horizon = horizon,
    goals = goals,
    environment = game_environment,
    observation_model = observation_model,
    observation_dim = observation_dim,
    state_dim = state_dim,
    action_dim = action_dim,
    game_structure = game_structure,
    )
end
function rotate_point(theta, point)
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    R * point
end

function pull_trajectory(recording; dir = "../../data/", track = [1, 2, 3], all = false, downsample_rate = 1, fill_traj = false)
    scale_down_factor = 10
    file = CSV.File(dir*recording*"_tracks.csv")
    raw_trajectories = (all) ? [[] for _ in 1:max(file[:trackId]...)+1] : [[] for _ in eachindex(track)]
    data_to_pull = [:xCenter, :yCenter, :heading, :xVelocity, :yVelocity, :xAcceleration, :yAcceleration, :width, :length,]
    for row in file
        idx = (all) ? row.trackId+1 : findfirst(x -> x == row.trackId, track)
        if !isnothing(idx)
            raw_state = [row[i] for i in data_to_pull]
            full_state = [
                scale_down_factor * rotate_point(2.303611, raw_state[1:2]) # + [390.5, 585.5]/10
                scale_down_factor * norm(raw_state[4:5])
                (deg2rad(raw_state[3]) + 5.445203653589793) % (2 * pi)
                ]
            push!(raw_trajectories[idx], full_state)
        end
    end

    traj = []
    # @infiltrate
    min_horizon = min([length(i) for i in raw_trajectories]...)
    max_horizon = max([length(i) for i in raw_trajectories]...)
    actual_horizon = fill_traj ? max_horizon : min_horizon
    for t in 1:actual_horizon 
        b = BlockVector(vcat([(t <= length(raw_trajectories[i])) ? raw_trajectories[i][t] : raw_trajectories[i][end] for i in eachindex(raw_trajectories)]...),
        [4 for _ in eachindex(raw_trajectories)])
        push!(traj, b)
    end
    return traj[1:downsample_rate:end]
end

"""
Structure of output is as follows:

 - [[x₁¹; x₂¹];[x₁²; x₂²];...;[x₁ᵀ; x₂ᵀ]] where xᵢʲ is the state of player i at time j in block vectors for each time

This structure is not consistent with MyopicSolver, do not pass in result to myopic solver.

"""
function reconstruct_solution(solution, game, horizon)
    num_player = num_players(game)
    player_state_dimension = convert(Int64, state_dim(game.dynamics)/num_player)

    primals = solution.primals
    solution = []
    for primal in primals
        vars = []
        for i in primal
            push!(vars, i)
        end
        push!(solution,vars)
    end

    player_states = [solution[i][1:player_state_dimension*horizon] for i in 1:num_player]
    solution = []
    for t in 1:horizon
        for i in 1:num_player
            push!(solution, 
                BlockVector(player_states[i][(t-1) * player_state_dimension + 1: t * player_state_dimension],
                    [player_state_dimension for _ in 1:num_player]
                )
            )
        end
    end
    solution
end
