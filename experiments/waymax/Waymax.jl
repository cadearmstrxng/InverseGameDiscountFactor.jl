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

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

function run_waymax_sim(;full_state=false, graph=true, verbose = true, myopic = true)

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
    
    # experiments/waymax/agent_states.txt
    agent_states = readlines("experiments/waymax/agent_states.txt")
    agent_states = [split(line, " ") for line in agent_states]
    agent_states = [[parse(Float64, x) for x in line] for line in agent_states]
    agent_states = [BlockArray(state, (4, 4, 4, 4)) for state in agent_states]
    
    initial_state = agent_states[1]
    goal_init_guess = [mean(roadgraph_points), mean(roadgraph_points)]
    if myopic
        game_params = mortar([
            [[goal_init_guess..., 1.0, 1.0, 1.0, 1.0, 1.0, 10.0 ] for i in 1:4]...])
    else
        game_params = mortar([
            [[goal_init_guess..., 1.0, 1.0, 1.0, 1.0, 10.0 ] for i in 1:4]...])
    end
    
    horizon = length(agent_states)

    init = init_waymax_test_game(;
        initial_state = initial_state,
        game_params = game_params,
        horizon = horizon,
        n = 4,
        dt = 0.04,
        myopic = myopic,
        verbose = verbose)

    !verbose || println("initial state: ", init.initial_state)
    !verbose || println("initial game parameters: ", init.game_parameters)
    !verbose || println("initial horizon: ", init.horizon)
    !verbose || println("observation model: ", init.observation_model)
    !verbose || println("observation dim: ", init.observation_dim)

    !verbose || println("game initialized\ninitializing mcp coupled optimization solver")
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )
    !verbose || println("mcp coupled optimization solver initialized")

    return function get_next_action_inverse(;)
        agent_states = readlines("experiments/waymax/agent_states.txt")
        agent_states = [split(line, " ") for line in agent_states]
        agent_states = [[parse(Float64, x) for x in line] for line in agent_states]
        agent_states = [BlockArray(state, (4, 4, 4, 4)) for state in agent_states]

        initial_state = agent_states[1]

        !verbose || println("solving inverse game")
        method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
            mcp_solver.mcp_game,
            agent_states,
            # forward_game_observations,
            init.observation_model,
            Tuple(blocksizes(init.game_parameters, 1));
            initial_state = init.initial_state,
            hidden_state_guess = init.game_parameters,
            max_grad_steps = 200,
            verbose = verbose,
            dynamics = dynamics,
            total_horizon = horizon,
            lr = 1e-4,
        )
        !verbose || println("finished inverse game")
        !verbose || println("recovered pararms: ", method_sol.recovered_params)

        recovered_params = method_sol.recovered_params

        current_state = agent_states[end]

        forward_solution = solve_mcp_game(
            mcp_solver.mcp_game,
            current_state,
            recovered_params
        )

        primals = forward_solution.primals
        player_state_dim = state_dim(dynamics.subsystems[1])
        player_control_dim = control_dim(dynamics.subsystems[1])
        ego_action_t1 = primals[player_state_dim+1:player_state_dim+player_control_dim]

        return ego_action_t1
        
    end
    
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

function init_waymax_test_game(;
    state_dim = (4, 4, 4, 4),
    action_dim = (2, 2, 2, 2),
    σ_ = 0.0,
    game_environment = nothing, 
    initial_state = mortar([
        [0, 2, 0.2236, 2*pi-1.10715], # initial x, y, initial velocity magnitude, heading angle (player 1)
        [2.5, 2, 0.0, 0.0],# player 2
        [2.5, 0, 0.0, 0.0] # player 3
    ]),
    game_params = mortar([
        [0.0, 0.0, 0.6, 1.0], # target x,y, discount, min_dist
        [0.0, 0.0, 0.6, 1.0],
        [0.0, 0.0, 0.6, 1.0]
    ]),
    horizon = 10,
    n = 4,
    dt = 0.04,
    myopic = false,
    verbose = false,
    lane_centers = nothing,
    dynamics = nothing,
)
    !verbose || print("initializing game ... ")
    game_structure = Waymax_collision_avoidance(
        n,
        lane_centers; # lane centers
        environment = game_environment,
        min_distance = 0.5,
        collision_avoidance_coefficient = 5.0,
        myopic = myopic,
        dynamics = dynamics
    )
    !verbose || print(" game structure initialized\n")
    
    observation_dim = state_dim[1]
    observation_model = 
        (x; σ = σ_) -> 
        BlockVector(
            vcat(
                [ x[state_dim[1] * (i - 1)+1:state_dim[1]*i] .+ σ * randn(state_dim[1])
                    for i in 1:n]...
            ),
            [state_dim[1] for _ in 1:n]
        )
    
    !verbose || println("observation model initialized")

    (;
    initial_state = initial_state,
    game_parameters = game_params,
    environment = game_environment,
    observation_model = observation_model,
    observation_dim = observation_dim,
    horizon = horizon,
    state_dim = state_dim,
    action_dim = action_dim,
    σ = σ_,
    game_structure = game_structure,
    )
end

function Waymax_collision_avoidance(
    num_players,
    lane_centers;
    environment,
    min_distance = 1.0, # context state 5
    collision_avoidance_coefficient = 20.0,
    dynamics = nothing,
    myopic = true)

    cost = let
        function target_cost(x, context_state, t)
            (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u, context_state, t)
            norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
        end
        function lane_center_cost(x, i, context_state, t)
            (myopic ? context_state[3] ^ t : 1) * norm_sqr(x[2] - lane_centers[i](x[1]))
        end
        function collision_cost(x, i, context_state, t)
            cost = [(1/(1+exp(10 * (norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]) - context_state[4])))) for paired_player in [1:(i - 1); (i + 1):num_players]]
            sum(cost)
        end
        function cost_for_player(i, xs, us, context_state, T)
            mean_target = mean([target_cost(xs[t + 1][Block(i)], context_state[Block(i)], t) for t in 1:T])
            control = mean([control_cost(us[t][Block(i)], context_state[Block(i)], t) for t in 1:T])
            safe_distance_violation = mean([collision_cost(xs[t], i, context_state[Block(i)], t) for t in 1:T])
            lane_center = mean([lane_center_cost(xs[t+1][Block(i)], i, context_state[Block(i)], t) for t in 1:T])

            context_state[Block(i)][myopic ? 5 : 4] * mean_target + 
            context_state[Block(i)][myopic ? 6 : 5] * control +
            context_state[Block(i)][myopic ? 7 : 6] * safe_distance_violation +
            (context_state[Block(i)][myopic ? 8 : 7] * lane_center : 0.0)

        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            T = size(us,1)
            [cost_for_player(i, xs, us, context_state, T) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    game = TrajectoryGame(
        dynamics,
        cost,
        environment,
        # shared_collision_avoidance_coupling_constraints(num_players, min_distance),
        nothing
    )
    InverseGameDiscountFactor.CollisionAvoidanceGame(game)

end

end 