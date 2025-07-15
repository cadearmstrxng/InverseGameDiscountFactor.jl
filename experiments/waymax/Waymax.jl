module Waymax

using BlockArrays
using TrajectoryGamesBase
using TrajectoryGamesExamples: BicycleDynamics
using LinearAlgebra
using Statistics: mean,median
using Random
using Infiltrator
using Optim

# include("../../src/InverseGameDiscountFactor.jl")
using InverseGameDiscountFactor

export run_waymax_sim, test_waymax_sim

struct CollisionAvoidanceGame
    game::TrajectoryGame
end

function run_waymax_sim(initial_agent_states;full_state=false, graph=false, verbose = false, myopic = true)

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

    ego_lane_center = x -> ego_model(x, ego_params)
    agent_3_lane_center = x -> agent_3_model(x, agent_3_params)
    agent_8_lane_center = x -> agent_8_point[2]
    agent_2_lane_center = x -> agent_2_point[2]

    lane_centers = [ego_lane_center, agent_3_lane_center, agent_8_lane_center, agent_2_lane_center]

    dynamics = BicycleDynamics(;
        dt = 0.1, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )
    
    agent_states_float = [[Float64(x) for x in line] for line in initial_agent_states]
    agent_states = [BlockArray(state, [4, 4, 4, 4]) for state in agent_states_float]
    
    initial_state = agent_states[1]
    goal_init_guess = mean(roadgraph_points)
    if myopic
        game_params = mortar([[vcat(goal_init_guess, [1.0, 1.0, 1.0, 1.0, 1.0, 10.0]) for i in 1:4]...])
    else
        game_params = mortar([[vcat(goal_init_guess, [1.0, 1.0, 1.0, 1.0, 10.0]) for i in 1:4]...])
    end
    
    horizon = length(agent_states)

    init = init_waymax_test_game(;
        initial_state = initial_state,
        game_params = game_params,
        horizon = horizon,
        n = 4,
        dt = 0.04,
        myopic = myopic,
        verbose = verbose,
        lane_centers = lane_centers
    )

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

    return function get_next_action_inverse(current_agent_states;)
        agent_states_float = [[Float64(x) for x in line] for line in current_agent_states]
        agent_states = [BlockArray(state, [4, 4, 4, 4]) for state in agent_states_float]

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
        player_state_dim = state_dim(dynamics)
        player_control_dim = control_dim(dynamics)
        ego_action_t1 = primals[1][player_state_dim*horizon+1:player_state_dim*horizon+player_control_dim]
        return ego_action_t1
    end
end

function calculate_road_func(roadpoints; display_plot = false)
    xs = map(p -> p[1], roadpoints)
    ys = map(p -> p[2], roadpoints)

    model(x, p) = p[1] * exp(p[2] * (x - p[3])) + p[4]
    
    loss(p) = sum(((x->model(x,p)).(xs) .- ys).^2)

    p0 = [1.0, 1.0, median(xs), median(ys)]

    lower = [-Inf, -100.0, -Inf, -Inf]
    upper = [Inf, 100.0, Inf, Inf]

    try
        result = optimize(loss, lower, upper, p0, Fminbox(LBFGS()))
        best_params = Optim.minimizer(result)
        println("Fit complete. Best parameters [c1, c2, c3]: ", best_params)

        # if display_plot
        #     CairoMakie.activate!()
        #     fig = Figure()
        #     ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel="x", ylabel="y", title="Exponential Fit with Optim.jl: y = c1*e^(c2*x) + c3")
            
        #     scatter!(ax, xs, ys, label="Data", markersize=4)
            
        #     x_smooth = range(minimum(xs), stop=maximum(xs), length=500)
        #     y_fit = model.(x_smooth, Ref(best_params))
        #     lines!(ax, x_smooth, y_fit, label="Fitted Curve", color=:red, linewidth=2)
            
        #     display(fig)
        # end
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
    game_environment = InverseGameDiscountFactor.NullEnv(), 
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
    dynamics = BicycleDynamics(;
        dt = 0.1, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    ),
)
    !verbose || print("initializing game ... ")
    game_structure = Waymax_collision_avoidance(
        n,
        lane_centers; # lane centers
        environment = game_environment,
        min_distance = 0.01,
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
            (context_state[Block(i)][myopic ? 8 : 7] * lane_center)

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

function norm_sqr(x)
    return sum(x.^2)
end

function test_waymax_sim(;myopic = true, verbose = false)
    initial_agent_states = [[2546.6337890625, -5539.427734375, 0.0003866358892992139, -0.27092814445495605, 2556.513671875, -5469.21435546875, 9.868803024291992, -1.5984519720077515, 2556.086669921875, -5521.36865234375, 11.002120971679688, -1.585107445716858, 2556.8125, -5497.447265625, 10.045750617980957, -1.5922516584396362], [2546.634033203125, -5539.427734375, 0.0006524321506731212, -0.27092671394348145, 2556.47216796875, -5470.30126953125, 10.87706184387207, -1.5992517471313477, 2556.066162109375, -5522.443359375, 10.749027252197266, -1.58596670627594, 2556.79150390625, -5498.45068359375, 10.036375999450684, -1.5920318365097046], [2546.6337890625, -5539.427734375, 0.0006872346857562661, -0.27092742919921875, 2556.423828125, -5471.35791015625, 10.577457427978516, -1.5981979370117188, 2556.05810546875, -5523.5302734375, 10.869439125061035, -1.5899896621704102, 2556.773681640625, -5499.4638671875, 10.133402824401855, -1.5921348333358765], [2546.6337890625, -5539.427734375, 0.000636950193438679, -0.27092742919921875, 2556.39013671875, -5472.4365234375, 10.791393280029297, -1.606227993965149, 2556.0361328125, -5524.60205078125, 10.720025062561035, -1.5878682136535645, 2556.751220703125, -5500.46630859375, 10.02692985534668, -1.592295527458191], [2546.6337890625, -5539.427734375, 0.0007294925744645298, -0.27092599868774414, 2556.359375, -5473.59423828125, 11.5812349319458, -1.5999959707260132, 2556.01806640625, -5525.66259765625, 10.607007026672363, -1.5875979661941528, 2556.7314453125, -5501.4833984375, 10.172821044921875, -1.5953866243362427], [2546.634033203125, -5539.427734375, 0.0007670021150261164, -0.27092552185058594, 2556.30712890625, -5474.703125, 11.101168632507324, -1.6102416515350342, 2555.992919921875, -5526.73876953125, 10.764656066894531, -1.5882734060287476, 2556.70947265625, -5502.46826171875, 9.851083755493164, -1.593716025352478], [2546.634033203125, -5539.427734375, 0.00023029708245303482, -0.27092552185058594, 2556.2646484375, -5475.791015625, 10.887197494506836, -1.6036913394927979, 2555.97216796875, -5527.8271484375, 10.885766983032227, -1.5893621444702148, 2556.689453125, -5503.47216796875, 10.041058540344238, -1.594173789024353], [2546.634033203125, -5539.42724609375, 0.00046955846482887864, -0.27092552185058594, 2556.217529296875, -5476.90234375, 11.123266220092773, -1.6065565347671509, 2555.946044921875, -5528.89892578125, 10.720956802368164, -1.5909461975097656, 2556.66796875, -5504.46826171875, 9.96325397491455, -1.5938026905059814], [2546.634033203125, -5539.427734375, 0.00025124350213445723, -0.27092552185058594, 2556.188232421875, -5477.97509765625, 10.731538772583008, -1.609052300453186, 2555.9208984375, -5529.9619140625, 10.632856369018555, -1.5908000469207764, 2556.646240234375, -5505.4521484375, 9.841265678405762, -1.594211220741272], [2546.634033203125, -5539.42724609375, 0.0008293227292597294, -0.27092552185058594, 2556.1494140625, -5479.041015625, 10.666245460510254, -1.6050068140029907, 2555.892822265625, -5531.037109375, 10.75561809539795, -1.5923048257827759, 2556.62255859375, -5506.44970703125, 9.97839641571045, -1.5917409658432007], [2546.634033203125, -5539.42724609375, 0.0004170200845692307, -0.27092552185058594, 2556.1259765625, -5480.21728515625, 11.765029907226562, -1.603785753250122, 2555.867919921875, -5532.11181640625, 10.749955177307129, -1.5928477048873901, 2556.602294921875, -5507.41455078125, 9.650565147399902, -1.5928443670272827]]
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

    ego_lane_center = x -> ego_model(x, ego_params)
    agent_3_lane_center = x -> agent_3_model(x, agent_3_params)
    agent_8_lane_center = x -> agent_8_point[2]
    agent_2_lane_center = x -> agent_2_point[2]

    lane_centers = [ego_lane_center, agent_3_lane_center, agent_8_lane_center, agent_2_lane_center]
    
    agent_states_float = [[Float64.(x) for x in line] for line in initial_agent_states]
    agent_states = [BlockArray(state, [4, 4, 4, 4]) for state in agent_states_float]
    
    initial_state = agent_states[1]
    goal_init_guess = mean(roadgraph_points)
    if myopic
        game_params = mortar([[vcat(goal_init_guess, [1.0, 1.0, 1.0, 1.0, 1.0, 10.0]) for i in 1:4]...])
    else
        game_params = mortar([[vcat(goal_init_guess, [1.0, 1.0, 1.0, 1.0, 10.0]) for i in 1:4]...])
    end
    
    horizon = length(agent_states)

    init = init_waymax_test_game(;
        initial_state = initial_state,
        game_params = game_params,
        horizon = horizon,
        n = 4,
        dt = 0.04,
        myopic = myopic,
        verbose = verbose,
        lane_centers = lane_centers
    )

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

    sol = solve_mcp_game(
        mcp_solver.mcp_game,
        initial_state,
        init.game_parameters;
        initial_guess = nothing,
        verbose = false,
        lr = 1e-3,
        total_horizon = 10,
        maxiter = 100000
    )
    @infiltrate
end
end 