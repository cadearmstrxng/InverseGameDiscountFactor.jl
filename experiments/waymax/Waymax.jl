module Waymax

using BlockArrays
using TrajectoryGamesBase
using TrajectoryGamesExamples
using LinearAlgebra
using Statistics: mean,median
using Random
using CairoMakie
using Optim

include("../GameUtils.jl")
using InverseGameDiscountFactor

export get_next_action, waymax_game, get_next_action_inverse


function run_waymax_sim(;full_state=false, graph=true, verbose = true)

    rng = MersenneTwister(1234)
    Random.seed!(rng)

    # Find Lane Center Funcs
    ego_roadgraph_idx = 3001:3340
    agent_3_roadgraph_idx = vcat(2000:2109, 2676:2900)
    agent_8_roadgraph_idx = [4441]
    agent_2_roadgraph_idx = agent_8_roadgraph_idx

    roadgraph_points = pull_roadpoints("experiments/waymax/data/roadgraph_points.txt")
    ego_roadgraph_points = roadgraph_points[ego_roadgraph_idx]
    agent_3_roadgraph_points = roadgraph_points[agent_3_roadgraph_idx]
    agent_8_roadgraph_points = roadgraph_points[agent_8_roadgraph_idx]
    agent_2_roadgraph_points = roadgraph_points[agent_2_roadgraph_idx]

    ego_model, ego_params = calculate_road_func(ego_roadgraph_points)
    ego_params[4] -= 2
    agent_3_model, agent_3_params = calculate_road_func(agent_3_roadgraph_points)
    agent_8_point = roadgraph_points[agent_8_roadgraph_idx][1]
    agent_2_point = roadgraph_points[agent_2_roadgraph_idx][1]

    ego_lane_center = x -> ego_model(x, Ref(ego_params))
    agent_3_lane_center = x -> agent_3_model(x, Ref(agent_3_params))
    agent_8_lane_center = x -> agent_8_point
    agent_2_lane_center = x -> agent_2_point

    lane_centers = [ego_lane_center, agent_3_lane_center, agent_8_lane_center, agent_2_lane_center]


    dynamics = TrajectoryGamesExamples.BicycleDynamics(;
        dt = 0.04*downsample_rate, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )


end

function get_next_action_inverse(
    current_state, 
    observed_trajectory; 
    horizon=10, 
    dt=0.1,
    game_params_guess = nothing,
    )
    num_players = length(current_state.blocks)

    dynamics = BicycleDynamics(;
        dt = dt,
        l = 2.8, # average car length
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]), # a, δ
    )

    game = waymax_game(num_players; dynamics=dynamics)
    
    # For the inverse solve, we need a guess for the parameters.
    # If not provided, we'll create a default one.
    if isnothing(game_params_guess)
        # Each player has [target_x, target_y, discount_factor]
        # We'll make each player's target their own final position in the observed trajectory.
        final_states = observed_trajectory[end].blocks
        game_params_guess_vec = [[fs[1], fs[2], 0.9] for fs in final_states]
        game_params_guess = BlockVector(mortar(game_params_guess_vec), [3 for _ in 1:num_players])
    end

    mcp_solver = MCPCoupledOptimizationSolver(game, horizon, blocksizes(game_params_guess, 1))

    # We need a simple observation model that just returns the state.
    # In this case, we assume full observability.
    observation_model(x) = x
    
    # Solve the inverse game to find the player parameters
    method_sol = solve_myopic_inverse_game(
        mcp_solver.mcp_game,
        observed_trajectory,
        observation_model,
        Tuple(blocksizes(game_params_guess, 1));
        initial_state = current_state,
        hidden_state_guess = game_params_guess,
        max_grad_steps = 100, # Keep it low for performance
        verbose = false,
        dynamics = dynamics,
        total_horizon = horizon,
        lr = 1e-3,
    )

    recovered_params = method_sol.recovered_params

    # Now, solve the forward game with the recovered parameters to get the action
    forward_solution = solve_mcp_game(
        mcp_solver.mcp_game,
        current_state,
        recovered_params
    )
    
    primals = forward_solution.primals
    state_dims = [state_dim(game.dynamics.subsystems[i]) for i in 1:num_players]
    control_dims = [control_dim(game.dynamics.subsystems[i]) for i in 1:num_players]
    control_offset = sum(state_dims)
    
    # Get the action for the first player (the SDC)
    start_idx = control_offset + 1
    end_idx = control_offset + control_dims[1]
    
    ego_action_t1 = primals[start_idx:end_idx]

    return ego_action_t1
end

function get_next_action(current_state, player_params; horizon=10, dt=0.1)
    num_players = length(current_state.blocks)

    dynamics = BicycleDynamics(;
        dt = dt,
        l = 2.8,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
    )

    game = waymax_game(num_players; dynamics=dynamics)

    mcp_solver = MCPCoupledOptimizationSolver(game, horizon, blocksizes(player_params, 1))
    
    solution = solve_mcp_game(
        mcp_solver.mcp_game,
        current_state,
        player_params
    )
    
    # The solution object holds the primal variables in `.primals`
    primals = solution.primals

    # Reconstruct the optimal control sequence from the primal variables
    state_dims = [state_dim(game.dynamics.subsystems[i]) for i in 1:num_players]
    control_dims = [control_dim(game.dynamics.subsystems[i]) for i in 1:num_players]
    
    # As per the problem formulation, variables are ordered [x1, ..., xN, u1, ..., uN] for each time step
    step_vars_count = sum(state_dims) + sum(control_dims)
    
    # We want the control for the first player at the first time step.
    # The offset for controls in a time step:
    control_offset = sum(state_dims)
    
    # The index for the first player's control at the first time step:
    start_idx = control_offset + 1
    end_idx = control_offset + control_dims[1]
    
    ego_action_t1 = primals[start_idx:end_idx]

    return ego_action_t1
end

function calculate_road_func(roadpoints; display_plot = false)
    xs = map(p -> p[1], roadpoints)
    ys = map(p -> p[2], roadpoints)

    model(x, p) = p[1] * exp(p[2] * (x - p[3])) + p[4]
    
    loss(p) = sum((model.(xs, Ref(p)) .- ys).^2)

    p0 = [1.0, 1.0, median(xs), median(ys)]

    lower = [-Inf, -100.0, -Inf, -Inf]
    upper = [Inf, 100.0, Inf, Inf]

    try
        result = optimize(loss, lower, upper, p0, Fminbox(LBFGS()))
        best_params = Optim.minimizer(result)
        println("Fit complete. Best parameters [c1, c2, c3]: ", best_params)

        if display_plot
            CairoMakie.activate!()
            fig = Figure()
            ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel="x", ylabel="y", title="Exponential Fit with Optim.jl: y = c1*e^(c2*x) + c3")
            
            scatter!(ax, xs, ys, label="Data", markersize=4)
            
            x_smooth = range(minimum(xs), stop=maximum(xs), length=500)
            y_fit = model.(x_smooth, Ref(best_params))
            lines!(ax, x_smooth, y_fit, label="Fitted Curve", color=:red, linewidth=2)
            
            display(fig)
        end
        return model, best_params
    catch e
        println("An error occurred during fitting: ", e)
    end
end

function pull_roadpoints(filename)
    data = readlines(filename)
    roadpoints = []
    for line in data
        x, y = split(line, " ")
        push!(roadpoints, [parse(Float64, x), parse(Float64, y)])
    end
    return roadpoints
end
end 