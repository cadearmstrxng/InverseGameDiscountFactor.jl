module InD

using BlockArrays: blocksizes, Block, mortar, BlockVector
using Infiltrator
using CairoMakie
using TrajectoryGamesBase:
    num_players, state_dim
using LinearAlgebra: norm_sqr
using Random
using ImageTransformations
using Rotations
using OffsetArrays:Origin
using TrajectoryGamesExamples: BicycleDynamics, PolygonEnvironment
using Statistics: mean, std

include("../GameUtils.jl")
include("../graphing/ExperimentGraphingUtils.jl")
include("../../src/InverseGameDiscountFactor.jl")
include("Environment.jl")

export run_bicycle_sim
function run_bicycle_sim(;full_state=true, graph=true, verbose = true)

    # InD_observations = GameUtils.observe_trajectory(forward_solution, init)
    frames = [26158, 26320] # 162
    # 201 25549, 26381
    # 205 25847,26381
    # 207,26098,26320
    # 208,26158,26381,
    tracks = [201, 205, 207, 208]
    downsample_rate = 9
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, downsample_rate = downsample_rate, all = false, frames = frames)
    open("InD_observations.tmp.txt", "w") do f
        for i in eachindex(InD_observations)
            write(f, string(round.(InD_observations[i]; digits = 4)), "\n")
        end
    end
    trk_201_lane_center(x) = 0.0  # Placeholder if no coefficients provided
    trk_205_lane_center(x) = 0.0  # Placeholder if no coefficients provided

    # 6th degree polynomial for track 207
    trk_207_lane_center(x) = -0.00017689424941367952*x^5 + 
    -0.01392776676762521*x^4 + 
    -0.38618068109105161*x^3 + 
    -3.938661796855388*x^2 + 
    1.9163167828141503*x + 
    252.1918028422481

    # Linear function for track 208
    trk_208_lane_center(x) = 8.304049624037807*x + 1.866183521575921e+02

    # Update the lane centers array to match the tracks array
    lane_centers = [trk_201_lane_center, trk_205_lane_center, trk_207_lane_center, trk_208_lane_center]
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate, # needs to become framerate
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 5.0] for i in 1:length(tracks)]...]),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = verbose,
        dynamics = dynamics,
        lane_centers = lane_centers,
        # game_environment = PolygonEnvironment(4, 200)
    )
    !verbose || println("initial state: ", init.initial_state)
    !verbose || println("initial game parameters: ", init.game_parameters)
    !verbose || println("initial horizon: ", init.horizon)
    !verbose || println("observation model: ", init.observation_model)
    !verbose || println("observation dim: ", init.observation_dim)
    
    InD_observations = (full_state) ? InD_observations : [BlockVector(GameUtils.observe_trajectory(InD_observations[t], init;blocked_by_time = false),
        [init.observation_dim for _ in 1:length(tracks)]) for t in 1:init.horizon]
    
    !verbose || println("game initialized\ninitializing mcp coupled optimization solver")
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )
    !verbose || println("mcp coupled optimization solver initialized")


    !verbose || println("solving inverse game")
    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_solver.mcp_game,
        InD_observations,
        # forward_game_observations,
        init.observation_model,
        Tuple(blocksizes(init.game_parameters, 1)),;
        initial_state = init.initial_state,
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        retries_on_divergence = 3,
        verbose = verbose,
        dynamics = dynamics,
    )
    !verbose || println("finished inverse game")
    !verbose || println("recovered pararms: ", method_sol.recovered_params)

    if graph
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered v. Warm Start",
            [InD_observations, method_sol.recovered_trajectory, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered",
            [InD_observations, method_sol.recovered_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Warm Start",
            [InD_observations, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)],
                [(:red, 0.2 ), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)]
            ],
            constraints = get_constraints(init.environment)
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Recovered v. Warm Start",
            [InD_observations, method_sol.recovered_trajectory, method_sol.warm_start_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 0.0), (:blue, 0.0), (:green, 0.0), (:purple, 0.0)],
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)], 
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)]
            ],
            constraints = get_constraints(init.environment)
        )
    end

    sol_error = norm_sqr(method_sol.recovered_trajectory - vcat(InD_observations...))
    !verbose || println("inverse sol error: ", sol_error)
    warm_sol_error = norm_sqr(method_sol.warm_start_trajectory - vcat(InD_observations...))
    !verbose || println("warm start sol error: ", warm_sol_error)
    !verbose || println("% improvement on warm start: ", (warm_sol_error - sol_error) / warm_sol_error * 100)
end

function compare_to_baseline(;full_state=true, graph=true, verbose = true)
    # InD_observations = GameUtils.observe_trajectory(forward_solution, init)
    frames = [26158, 26320] # 162
    # 201 25549, 26381
    # 205 25847,26381
    # 207,26098,26320
    # 208,26158,26381,
    tracks = [201, 205, 207, 208]
    downsample_rate = 6
    σ = 0.0
    rng = MersenneTwister(1234)
    Random.seed!(rng)

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
    first_full_observation = InD_observations[1]
    InD_observations = full_state ? 
        InD_observations :
        [BlockVector(
            mapreduce(x -> x[1:2], vcat, observation.blocks),
            [2 for _ in 1:length(tracks)]) 
        for observation in InD_observations]

    trk_201_lane_center(x) = 0.0  # Placeholder if no coefficients provided
    trk_205_lane_center(x) = 0.0  # Placeholder if no coefficients provided

    # 6th degree polynomial for track 207
    trk_207_lane_center(x) = -0.00017689424941367952*x^5 + 
    -0.01392776676762521*x^4 + 
    -0.38618068109105161*x^3 + 
    -3.938661796855388*x^2 + 
    1.9163167828141503*x + 
    252.1918028422481

    # Linear function for track 208
    trk_208_lane_center(x) = 8.304049624037807*x + 1.866183521575921e+02
    lane_centers = [trk_201_lane_center, trk_205_lane_center, trk_207_lane_center, trk_208_lane_center]
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate,
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    # Initialize game with full state observation
    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = first_full_observation,
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    init_baseline = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = init.initial_state,
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = init.horizon,
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=false,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )
    # Create MCP game solvers
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )

    baseline_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init_baseline.game_structure.game,
        init_baseline.horizon,
        [7 for _ in 1:length(tracks)]
    )

    noisy_observations = map(InD_observations) do obs
        obs .+ σ * randn(size(obs))
    end

    # Solve inverse game with both methods
    print("solving inverse game new method...")
    method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        mcp_solver.mcp_game,
        noisy_observations,
        init.observation_model,
        blocksizes(init.game_parameters, 1);
        initial_state = init.initial_state,
        hidden_state_guess = init.game_parameters,
        max_grad_steps = 200,
        verbose = false,
        dynamics = dynamics,
    )
    println("done")
    print("solving inverse game baseline method...")

    baseline_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
        baseline_solver.mcp_game,
        noisy_observations,
        init_baseline.observation_model,
        blocksizes(init_baseline.game_parameters, 1);
        initial_state = init_baseline.initial_state,
        hidden_state_guess = init_baseline.game_parameters,
        max_grad_steps = 200,
        verbose = false,
        dynamics = dynamics,
    )
    println("done")
    for i in 1:length(tracks)
        println("method sol: ", method_sol.recovered_params[(i-1)*blocksizes(init.game_parameters, 1)[1]+1:i*blocksizes(init.game_parameters, 1)[1]])
    end
    for i in 1:length(tracks)
        println("baseline sol: ", baseline_sol.recovered_params[(i-1)*blocksizes(init_baseline.game_parameters, 1)[1]+1:i*blocksizes(init_baseline.game_parameters, 1)[1]])
    end
    method_error = norm_sqr(method_sol.recovered_trajectory - vcat(InD_observations...))
    baseline_error = norm_sqr(baseline_sol.recovered_trajectory - vcat(InD_observations...))

    println("method error: ", method_error)
    println("baseline error: ", baseline_error)

    println("improvement: ", (baseline_error - method_error) / baseline_error * 100)

    println("method recovered params: ", method_sol.recovered_params)
    println("baseline recovered params: ", baseline_sol.recovered_params)

    if graph
        # Convert trajectories to consistent format for plotting
        # method_trajectory = BlockVector(
        #     [method_sol.recovered_trajectory[Block(i)][1:2] for i in 1:length(method_sol.recovered_trajectory.blocks)],
        #     [2 for _ in 1:length(tracks)]
        # )
        
        # baseline_trajectory = BlockVector(
        #     [baseline_sol.recovered_trajectory[Block(i)][1:2] for i in 1:length(baseline_sol.recovered_trajectory.blocks)],
        #     [2 for _ in 1:length(tracks)]
        # )
        method_trajectory = BlockVector(
            method_sol.recovered_trajectory,
            [init.observation_dim*length(tracks) for _ in 1:init.horizon]
        )
        baseline_trajectory = BlockVector(
            baseline_sol.recovered_trajectory,
            [init.observation_dim*length(tracks) for _ in 1:init.horizon]
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Observed v. Recovered v. Baseline",
            [InD_observations, method_trajectory, baseline_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 0.3), (:blue, 0.3), (:green, 0.3), (:purple, 0.3)],
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.7), (:blue, 0.7), (:green, 0.7), (:purple, 0.7)]
            ],
            constraints = init.environment === nothing ? nothing : get_constraints(init.environment),
            p_state_dim = 2
        )

        ExperimentGraphingUtils.graph_trajectories(
            "Recovered v. Baseline",
            [InD_observations, method_trajectory, baseline_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 0.0), (:blue, 0.0), (:green, 0.0), (:purple, 0.0)],
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)]
            ],
            constraints = init.environment === nothing ? nothing : get_constraints(init.environment),
            p_state_dim = 2
        )
        ExperimentGraphingUtils.graph_trajectories(
            "Recovered v. Observed",
            [InD_observations, method_trajectory, baseline_trajectory],
            init.game_structure,
            init.horizon;
            colors = [
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
                [(:red, 0.5), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)],
                [(:red, 0.0), (:blue, 0.0), (:green, 0.0), (:purple, 0.0)]
            ],
            constraints = init.environment === nothing ? nothing : get_constraints(init.environment),
            p_state_dim = 2
        )
        open("solved_method_trajectory_$(σ).txt", "w") do f
            for state in method_trajectory.blocks
                write(f, "-------------------------\n")
                for i in 1:length(tracks)
                    write(f, string(round.(state[(i-1)*4 + 1:i*4]; digits = 4)), "\n")
                end
            end
            write(f, "-------------------------\n")
        end
        open("baseline_method_trajectory_$(σ).txt", "w") do f
            for state in baseline_trajectory.blocks
                write(f, "-------------------------\n")
                for i in 1:length(tracks)
                    write(f, string(round.(state[(i-1)*4 + 1:i*4]; digits = 4)), "\n")
                end
            end
            write(f, "-------------------------\n")
        end
    end
end

function compare_noise_levels(;full_state=true, noise_levels=[0.0, 0.01, 0.05, 0.1], verbose=true)
    # Same trajectory data as other functions
    frames = [26158, 26320] # 162
    tracks = [201, 205, 207, 208]
    downsample_rate = 6
    rng = MersenneTwister(1234)
    Random.seed!(rng)

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
    InD_observations = full_state ? InD_observations : [observation[1:2] for observation in InD_observations]
    
    # Lane center functions
    trk_201_lane_center(x) = 0.0
    trk_205_lane_center(x) = 0.0
    trk_207_lane_center(x) = -6.535465682649165e-04*x^6 + 
                            -0.069559792458210*x^5 + 
                            -3.033950160533982*x^4 + 
                            -69.369975733866840*x^3 + 
                            -8.760325006936075e+02*x^2 + 
                            -5.782944928944775e+03*x + 
                            -1.547509969706588e+04
    trk_208_lane_center(x) = 8.304049624037807*x + 1.866183521575921e+02
    
    lane_centers = [trk_201_lane_center, trk_205_lane_center, trk_207_lane_center, trk_208_lane_center]
    
    # Setup dynamics
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate,
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    # Initialize base game with full state observation
    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    # Create MCP game solver
    mcp_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )

    # Storage for results
    all_solutions = []
    all_errors = []
    
    # Run experiment for each noise level
    for σ in noise_levels
        if verbose
            println("Running with noise level σ = ", σ)
        end
        
        # Apply noise to observations
        noisy_observations = map(InD_observations) do obs
            BlockVector(init.observation_model(obs, σ=σ),
                [Int64(state_dim(init.game_structure.game.dynamics) ÷ num_players(init.game_structure.game)) 
                for _ in 1:num_players(init.game_structure.game)])
        end
        
        # Solve inverse game with the noisy observations
        method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
            mcp_solver.mcp_game,
            noisy_observations,
            init.observation_model,
            blocksizes(init.game_parameters, 1);
            initial_state = init.initial_state,
            hidden_state_guess = init.game_parameters,
            max_grad_steps = 200,
            verbose = verbose,
            dynamics = dynamics,
        )
        
        # Calculate error
        error = norm_sqr(method_sol.recovered_trajectory - vcat(InD_observations...))
        
        push!(all_solutions, method_sol)
        push!(all_errors, error)
        
        if verbose
            println("Error at noise level σ = ", σ, ": ", error)
            for i in 1:length(tracks)
                block_size = blocksizes(init.game_parameters, 1)[1]
                recovered_params = method_sol.recovered_params[(i-1)*block_size+1:i*block_size]
                println("Player ", i, " recovered params: ", recovered_params)
            end
        end

        player_state_dim = state_dim(init.game_structure.game.dynamics) ÷ num_players(init.game_structure.game)

        # Write the solved trajectory to a text file
        open("solved_trajectory_$(σ).txt", "w") do f
            for state in method_sol.recovered_trajectory.blocks
                for i in 1:length(tracks)
                    write(f, string(round.(state[(i-1)*player_state_dim + 1:i*player_state_dim]; digits = 4)), "\n")
                end
                write(f, "--------------------------------\n")
            end
        end

        if verbose
            println("Solved trajectory for noise level σ = ", σ, " written to solved_trajectory_$(σ).txt")
        end
    end
end

function receding_horizon_snapshots(;full_state=true, graph=true, verbose = true)
    frames = [26158, 26320] # 162
    tracks = [201, 205, 207, 208]
    downsample_rate = 8
    rng = MersenneTwister(1234)
    Random.seed!(rng)

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
    total_horizon = length(frames[1]:downsample_rate:frames[2])
    
    # Lane center functions
    trk_201_lane_center(x) = 0.0
    trk_205_lane_center(x) = 0.0
    trk_207_lane_center(x) = -6.535465682649165e-04*x^6 + 
                            -0.069559792458210*x^5 + 
                            -3.033950160533982*x^4 + 
                            -69.369975733866840*x^3 + 
                            -8.760325006936075e+02*x^2 + 
                            -5.782944928944775e+03*x + 
                            -1.547509969706588e+04
    trk_208_lane_center(x) = 8.304049624037807*x + 1.866183521575921e+02
    
    lane_centers = [trk_201_lane_center, trk_205_lane_center, trk_207_lane_center, trk_208_lane_center]
    
    # Setup dynamics
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate,
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    init_rh = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = 10,
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    # baseline_init_rh = GameUtils.init_bicycle_test_game(
    #     full_state;
    #     initial_state = InD_observations[1],
    #     game_params = mortar([
    #         [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
    #     horizon = 10,
    #     n = length(tracks),
    #     dt = 0.04*downsample_rate,
    #     myopic=false,
    #     verbose = false,
    #     dynamics = dynamics,
    #     lane_centers = lane_centers
    # )

    solver_rh = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init_rh.game_structure.game,
        init_rh.horizon,
        blocksizes(init_rh.game_parameters, 1)
    )

    # baseline_solver_rh = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
    #     baseline_init_rh.game_structure.game,
    #     baseline_init_rh.horizon,
    #     blocksizes(baseline_init_rh.game_parameters, 1)
    # )

    # rh_plans = []
    # baseline_rh_plans = []
    inverse_costs_future = []
    baseline_inverse_costs_future = []
    params_future = []
    baseline_params_future = []
    actual_traj = []
    baseline_actual_traj = []
    initial_state_guess = init_rh.initial_state
    initial_hidden_state_guess = init_rh.game_parameters
    @infiltrate
    for t in 10:total_horizon-10
        rh_observations = InD_observations[t-10+1:t]
        # @infiltrate

        inverse_rh = InverseGameDiscountFactor.solve_myopic_inverse_game(
            solver_rh.mcp_game,
            rh_observations,
            init_rh.observation_model,
            blocksizes(init_rh.game_parameters, 1);
            initial_state = initial_state_guess,
            hidden_state_guess = nothing,
            max_grad_steps = 200,
            verbose = verbose,
            dynamics = dynamics,
            use_warm_start = false
        )

        initial_hidden_state_guess = inverse_rh.recovered_params

        # baseline_inverse_rh = InverseGameDiscountFactor.solve_myopic_inverse_game(
        #     baseline_solver_rh.mcp_game,
        #     rh_observations,
        #     baseline_init_rh.observation_model,
        #     blocksizes(baseline_init_rh.game_parameters, 1);
        #     initial_state = baseline_init_rh.initial_state,
        #     hidden_state_guess = baseline_init_rh.game_parameters,
        #     max_grad_steps = 200,
        #     verbose = verbose,
        #     dynamics = dynamics,
        #     use_warm_start = false
        # )     

        rh_initial_state = BlockVector(inverse_rh.recovered_trajectory.blocks[end], [4 for _ in 1:num_players(init_rh.game_structure.game)])
        # baseline_rh_initial_state = BlockVector(baseline_inverse_rh.recovered_trajectory.blocks[end], [4 for _ in 1:num_players(baseline_init_rh.game_structure.game)])
        println("solving forward game $t....")
        rh_plan = InverseGameDiscountFactor.reconstruct_solution(
            # InverseGameDiscountFactor.solve_mcp_game(solver_rh.mcp_game, rh_observations[end], inverse_rh.recovered_params), 
            InverseGameDiscountFactor.solve_mcp_game(solver_rh.mcp_game, rh_initial_state, inverse_rh.recovered_params), 
            init_rh.game_structure.game, 
            init_rh.horizon
        )
        println("forward game solved")
        # initial_state_guess = rh_initial_state # maybe change
        
        # baseline_rh_plan = InverseGameDiscountFactor.reconstruct_solution(
        #     # InverseGameDiscountFactor.solve_mcp_game(baseline_solver_rh.mcp_game, rh_observations[end], baseline_inverse_rh.recovered_params), 
        #     InverseGameDiscountFactor.solve_mcp_game(baseline_solver_rh.mcp_game, baseline_rh_initial_state, baseline_inverse_rh.recovered_params), 
        #     baseline_init_rh.game_structure.game, 
        #     baseline_init_rh.horizon
        # )
        if t == 18
            push!(actual_traj, rh_plan)
            # push!(baseline_actual_traj, baseline_rh_plan)
        else
            if t == 10
                push!(actual_traj, inverse_rh.recovered_trajectory)
                # push!(baseline_actual_traj, baseline_inverse_rh.recovered_trajectory)
            end
            push!(actual_traj, rh_plan.blocks[1])
            # push!(baseline_actual_traj, baseline_rh_plan.blocks[1])
        end

        # if graph
        #     ExperimentGraphingUtils.graph_rh_snapshot(
        #         "rh_snapshot_t$(t)",
        #         rh_observations,
        #         inverse_rh.recovered_trajectory,
        #         baseline_inverse_rh.recovered_trajectory,
        #         rh_plan,
        #         baseline_rh_plan,
        #         init_rh.game_structure,
        #         init_rh.horizon
        #     )
        # end

    #     push!(inverse_costs_future, norm(vcat(InD_observations[t+1:t+10]...) - rh_plan))
    #     push!(baseline_inverse_costs_future, norm(vcat(InD_observations[t+1:t+10]...) - baseline_rh_plan))
    #     push!(params_future, inverse_rh.recovered_params)
    #     push!(baseline_params_future, baseline_inverse_rh.recovered_params)
    #     open("rh.txt", "a") do f
    #         write(f, "t: ", string(t), "\n")
    #         write(f, "inverse_costs_future: ", string(round.(inverse_costs_future[end]; digits = 4)), "\n")
    #         write(f, "baseline_inverse_costs_future: ", string(round.(baseline_inverse_costs_future[end]; digits = 4)), "\n")
    #         write(f, "params_future: ", string(round.(params_future[end]; digits = 4)), "\n")
    #         write(f, "baseline_params_future: ", string(round.(baseline_params_future[end]; digits = 4)), "\n")
    #         open("baseline_rh_plan_$(t).txt", "w") do f
    #             for state in baseline_rh_plan.blocks
    #                 for i in 1:length(tracks)
    #                     write(f, string(round.(state[(i-1)*4 + 1:i*4]; digits = 4)), "\n")
    #                 end
    #                 write(f, "--------------------------------\n")
    #             end
    #         end
    #         open("rh_plan_$(t).txt", "w") do f
    #             for state in rh_plan.blocks
    #                 for i in 1:length(tracks)
    #                     write(f, string(round.(state[(i-1)*4 + 1:i*4]; digits = 4)), "\n")
    #                 end
    #                 write(f, "--------------------------------\n")
    #             end
    #         end
    #     end
    end
    actual_traj = BlockVector(vcat(actual_traj...), [16 for _ in 1:total_horizon])
    # baseline_actual_traj = BlockVector(vcat(baseline_actual_traj...), [16 for _ in 1:total_horizon])


    

    open("whole_trajectory.txt", "w") do f
        for state in actual_traj.blocks
            for i in 1:length(tracks)
                write(f, string(round.(state[(i-1)*4 + 1:i*4]; digits = 4)), "\n")
            end
            write(f, "--------------------------------\n")
        end
    end
    # open("baseline_whole_trajectory.txt", "w") do f
    #     for state in baseline_actual_traj.blocks
    #         for i in 1:length(tracks)
    #             write(f, string(round.(state[(i-1)*4 + 1:i*4]; digits = 4)), "\n")
    #         end
    #         write(f, "--------------------------------\n")
    #     end
    # end

    ExperimentGraphingUtils.graph_trajectories(
            "Recovered v. Observed",
            [InD_observations, actual_traj],
            init_rh.game_structure,
            28;
            colors = [
                [(:red, 0.0), (:blue, 0.0), (:green, 0.0), (:purple, 0.0)],
                [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
            ],
            constraints = nothing,
            p_state_dim = 4
        )
    # open("rh.txt", "w") do f
    #     for time in 10:total_horizon-10
    #         t = time - 9
    #         write(f, "t: ", string(t), "\n")
    #         write(f, "inverse_costs_future: ", string(round.(inverse_costs_future[t]; digits = 4)), "\n")
    #         write(f, "baseline_inverse_costs_future: ", string(round.(baseline_inverse_costs_future[t]; digits = 4)), "\n")
    #         write(f, "params_future: ", string(round.(params_future[t]; digits = 4)), "\n")
    #         write(f, "baseline_params_future: ", string(round.(baseline_params_future[t]; digits = 4)), "\n")
    #     end
    # end
end

function plot_rh()
    fig1 = ExperimentGraphingUtils.plot_rh_costs("./experiments/In-D/rh_snapshot/rh.txt")
    frames = [26158, 26320] # 162
    tracks = [201, 205, 207, 208]
    downsample_rate = 6

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
    final_obs = InD_observations[end]
    
    true_params = zeros(32)  # 8 parameters per player, 4 players
    
    # Set goal positions for each player (first two parameters per player)
    for i in 1:4
        true_params[(i-1)*8+1] = final_obs[Block(i)][1]  # x position
        true_params[(i-1)*8+2] = final_obs[Block(i)][2]  # y position
        # Set default values for other parameters
        true_params[(i-1)*8+3] = 1.0  # discount factor
        true_params[(i-1)*8+4] = 1.0  # other parameters
        true_params[(i-1)*8+5] = 1.0
        true_params[(i-1)*8+6] = 1.0
        true_params[(i-1)*8+7] = 1.0
        true_params[(i-1)*8+8] = 10.0
    end
    
    # Generate the parameter difference plots
    fig2 = ExperimentGraphingUtils.plot_rh_parameter_differences("./experiments/In-D/rh_snapshot/rh.txt", true_params)
    
    # Parse the rh.txt file to get parameter histories
    times, _, _, params_history, baseline_params_history = ExperimentGraphingUtils.parse_rh_file("./experiments/In-D/rh_snapshot/rh.txt")
    
    # Generate the goal estimates plot
    ExperimentGraphingUtils.plot_goal_estimates(
        "./experiments/In-D/rh_snapshot/goal_estimates",
        params_history,
        baseline_params_history,
        true_params
    )
    
    println("Plotting complete. Files saved:")
    println("  - costs over time.pdf")
    println("  - parameter differences over time.pdf")
    println("  - parameter differences per player over time.pdf")
    println("  - goal differences per player over time.pdf")
    println("  - goal_estimates.png")
end

function generate_visualization()
    CairoMakie.activate!();
    fig = CairoMakie.Figure()
    image_data = CairoMakie.load("experiments/data/07_background.png")
    image_data = image_data[end:-1:1, :]
    image_data = image_data'
    ax1 = Axis(fig[1,1], aspect = DataAspect())
    trfm = ImageTransformations.recenter(Rotations.RotMatrix(-2.303611),center(image_data))

    x_crop_min = 430
    x_crop_max = 875
    y_crop_min = 225
    y_crop_max = 1025
    
    scale = 1/10.25

    x = (x_crop_max - x_crop_min) * scale
    y = (y_crop_max - y_crop_min) * scale

    # println(x,' ', y)

    image_data = ImageTransformations.warp(image_data, trfm)
    image_data = Origin(0)(image_data)
    image_data = image_data[x_crop_min:x_crop_max, y_crop_min:y_crop_max]
    
    x_offset = -34.75
    y_offset = 22

    # println(x_offset..(x+x_offset-2), ' ', y_offset..(y-2+y_offset))

    image!(ax1,
        x_offset..(x+x_offset),
        y_offset..(y+y_offset),
        image_data)

    num_players = 4
    horizon = 28

    # Get real data
    frames = [26158, 26320]
    tracks = [201, 205, 207, 208]
    downsample_rate = 6
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, downsample_rate = downsample_rate, all = false, frames = frames)

    InD_observations = let 
        new_observations = []
        for observation_t in InD_observations
            # states = [observation_t[Block(i)] for i in 1:num_players]
            for i in 1:num_players
                push!(new_observations, observation_t[Block(i)])
            end
            # push!(new_observations, observation_t[Block(i)] for i in 1:num_players)
        end
        BlockVector(new_observations, [num_players for _ in 1:horizon])
    end

    # Get trajectories
    t1 = open("solved_trajectory_0.0.txt") do f
        t = readlines(f)
        new_t = []
        for line in t
            if line == "--------------------------------"
                continue
            end
            push!(new_t, parse.(Float64, split(chop(line; head=1, tail=1), ',')))
        end
        BlockVector(new_t, [num_players for _ in 1:horizon])
    end

    t2 = open("solved_trajectory_0.1.txt") do f
        t = readlines(f)
        new_t = []
        for line in t
            if line == "--------------------------------"
                continue
            end
            push!(new_t, parse.(Float64, split(chop(line; head=1, tail=1), ',')))
        end
        BlockVector(new_t, [num_players for _ in 1:horizon])
    end

    t3 = open("solved_trajectory_0.01.txt") do f
        t = readlines(f)
        new_t = []
        for line in t
            if line == "--------------------------------"
                continue
            end
            push!(new_t, parse.(Float64, split(chop(line; head=1, tail=1), ',')))
        end
        BlockVector(new_t, [num_players for _ in 1:horizon])
    end

    t4 = open("solved_trajectory_0.05.txt") do f
        t = readlines(f)
        new_t = []
        for line in t
            if line == "--------------------------------"
                continue
            end
            push!(new_t, parse.(Float64, split(chop(line; head=1, tail=1), ',')))
        end
        BlockVector(new_t, [num_players for _ in 1:horizon])
    end

    # Get position data

    InD_observations = BlockVector([observation[1:2] for observation in InD_observations], [num_players for _ in 1:horizon])
    trajectory = BlockVector([t[1:2] for t in t1], [num_players for _ in 1:horizon])

    # Add noise to observations

    σ = 0.01
    state_dim = 2

    observation_model = 
            (obs; σ = σ_) -> 
                [ x .+ σ * randn(state_dim)
                    for x in obs]
            

    noisy_observations = observation_model(InD_observations, σ=σ)

    # Plot Data

    Observations_x_player_1 = [noisy_observations[Block(t)][1][1] for t in 1:horizon]
    Observations_y_player_1 = [noisy_observations[Block(t)][1][2] for t in 1:horizon]
    Observations_x_player_2 = [noisy_observations[Block(t)][2][1] for t in 1:horizon]
    Observations_y_player_2 = [noisy_observations[Block(t)][2][2] for t in 1:horizon]
    Observations_x_player_3 = [noisy_observations[Block(t)][3][1] for t in 1:horizon]
    Observations_y_player_3 = [noisy_observations[Block(t)][3][2] for t in 1:horizon]
    Observations_x_player_4 = [noisy_observations[Block(t)][4][1] for t in 1:horizon]
    Observations_y_player_4 = [noisy_observations[Block(t)][4][2] for t in 1:horizon]
    Inverse_x_player_1 = [trajectory[Block(t)][1][1] for t in 1:horizon]
    Inverse_y_player_1 = [trajectory[Block(t)][1][2] for t in 1:horizon]
    Inverse_x_player_2 = [trajectory[Block(t)][2][1] for t in 1:horizon]
    Inverse_y_player_2 = [trajectory[Block(t)][2][2] for t in 1:horizon]
    Inverse_x_player_3 = [trajectory[Block(t)][3][1] for t in 1:horizon]
    Inverse_y_player_3 = [trajectory[Block(t)][3][2] for t in 1:horizon]
    Inverse_x_player_4 = [trajectory[Block(t)][4][1] for t in 1:horizon]
    Inverse_y_player_4 = [trajectory[Block(t)][4][2] for t in 1:horizon]

    p1_observations = scatter!(ax1, Observations_x_player_1, Observations_y_player_1, color = :red, markersize = 5, alpha = 0.5)
    p2_observations = scatter!(ax1, Observations_x_player_2, Observations_y_player_2, color = :blue, markersize = 5, alpha = 0.5)
    p3_observations = scatter!(ax1, Observations_x_player_3, Observations_y_player_3, color = :green, markersize = 5, alpha = 0.5)
    p4_observations = scatter!(ax1, Observations_x_player_4, Observations_y_player_4, color = :purple, markersize = 5, alpha = 0.5)

    p1_inverse = lines!(ax1, Inverse_x_player_1, Inverse_y_player_1, color = :red)
    p2_inverse = lines!(ax1, Inverse_x_player_2, Inverse_y_player_2, color = :blue)
    p3_inverse = lines!(ax1, Inverse_x_player_3, Inverse_y_player_3, color = :green)
    p4_inverse = lines!(ax1, Inverse_x_player_4, Inverse_y_player_4, color = :purple)

    Legend(fig[1,2],[
        p1_observations, 
        p2_observations, 
        p3_observations, 
        p4_observations, 
        p1_inverse, 
        p2_inverse, 
        p3_inverse, 
        p4_inverse
        ], [
        "Player 1 Observation",
        "Player 2 Observation",
        "Player 3 Observation",
        "Player 4 Observation",
        "Player 1 Inverse",
        "Player 2 Inverse",
        "Player 3 Inverse",
        "Player 4 Inverse"
    ])

    CairoMakie.save("InD_visualization", fig)
end

function monte_carlo_rh_study(;full_state=true, verbose=true)
    # Fixed parameters
    noise_level = 0.002
    num_trials = 50
    frames = [26158, 26320]
    tracks = [201, 205, 207, 208]
    downsample_rate = 6
    total_horizon = length(frames[1]:downsample_rate:frames[2])
    planning_horizon = 10  # Fixed planning horizon for RH
    
    # Initialize random number generator
    rng = MersenneTwister(1234)
    Random.seed!(rng)
    
    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
    
    # Lane center functions
    trk_201_lane_center(x) = 0.0
    trk_205_lane_center(x) = 0.0
    trk_207_lane_center(x) = -6.535465682649165e-04*x^6 + 
                            -0.069559792458210*x^5 + 
                            -3.033950160533982*x^4 + 
                            -69.369975733866840*x^3 + 
                            -8.760325006936075e+02*x^2 + 
                            -5.782944928944775e+03*x + 
                            -1.547509969706588e+04
    trk_208_lane_center(x) = 8.304049624037807*x + 1.866183521575921e+02
    
    lane_centers = [trk_201_lane_center, trk_205_lane_center, trk_207_lane_center, trk_208_lane_center]
    
    # Setup dynamics
    dynamics = BicycleDynamics(;
        dt = 0.04*downsample_rate,
        l = 1.0,
        state_bounds = (; lb = [-Inf, -Inf, -Inf, -Inf], ub = [Inf, Inf, Inf, Inf]),
        control_bounds = (; lb = [-5, -pi/4], ub = [5, pi/4]),
        integration_scheme = :forward_euler
    )

    # Initialize base game with full state observation
    init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = planning_horizon,
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    baseline_init = GameUtils.init_bicycle_test_game(
        full_state;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 10.0] for i in 1:length(tracks)]...]),
        horizon = planning_horizon,
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=false,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    # Create MCP game solvers
    solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    )

    baseline_solver = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        baseline_init.game_structure.game,
        baseline_init.horizon,
        blocksizes(baseline_init.game_parameters, 1)
    )

    # Storage for results
    method_goal_errors = zeros(num_trials, total_horizon - 2*planning_horizon + 1)
    baseline_goal_errors = zeros(num_trials, total_horizon - 2*planning_horizon + 1)
    method_player_errors = zeros(num_trials, total_horizon - 2*planning_horizon + 1, length(tracks))
    baseline_player_errors = zeros(num_trials, total_horizon - 2*planning_horizon + 1, length(tracks))
    
    # Get true goal positions
    true_goals = zeros(8)  # 2 positions per player
    for i in 1:4
        true_goals[(i-1)*2+1] = InD_observations[end][Block(i)][1]
        true_goals[(i-1)*2+2] = InD_observations[end][Block(i)][2]
    end

    # Run Monte Carlo trials
    for trial in 1:num_trials
        if verbose
            println("Running trial ", trial, " of ", num_trials)
        end
        
        # Add noise to observations
        noisy_observations = map(InD_observations) do obs
            BlockVector(init.observation_model(obs, σ=noise_level),
                [Int64(state_dim(init.game_structure.game.dynamics) ÷ num_players(init.game_structure.game)) 
                for _ in 1:num_players(init.game_structure.game)])
        end

        # Run receding horizon for each time step
        for t in planning_horizon:total_horizon-planning_horizon
            rh_observations = noisy_observations[t-planning_horizon+1:t]
            
            # Solve with new method
            inverse_rh = InverseGameDiscountFactor.solve_myopic_inverse_game(
                solver.mcp_game,
                rh_observations,
                init.observation_model,
                blocksizes(init.game_parameters, 1);
                initial_state = init.initial_state,
                hidden_state_guess = init.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                dynamics = dynamics,
                use_warm_start = false
            )

            # Solve with baseline method
            baseline_inverse_rh = InverseGameDiscountFactor.solve_myopic_inverse_game(
                baseline_solver.mcp_game,
                rh_observations,
                baseline_init.observation_model,
                blocksizes(baseline_init.game_parameters, 1);
                initial_state = baseline_init.initial_state,
                hidden_state_guess = baseline_init.game_parameters,
                max_grad_steps = 200,
                verbose = false,
                dynamics = dynamics,
                use_warm_start = false
            )

            # Extract goal positions from recovered parameters
            method_goals = zeros(8)
            baseline_goals = zeros(8)
            
            for i in 1:4
                method_goals[(i-1)*2+1] = inverse_rh.recovered_params[(i-1)*8+1]
                method_goals[(i-1)*2+2] = inverse_rh.recovered_params[(i-1)*8+2]
                baseline_goals[(i-1)*2+1] = baseline_inverse_rh.recovered_params[(i-1)*7+1]
                baseline_goals[(i-1)*2+2] = baseline_inverse_rh.recovered_params[(i-1)*7+2]
                
                # Calculate per-player errors
                method_player_errors[trial, t-planning_horizon+1, i] = norm(method_goals[(i-1)*2+1:i*2] - true_goals[(i-1)*2+1:i*2])
                baseline_player_errors[trial, t-planning_horizon+1, i] = norm(baseline_goals[(i-1)*2+1:i*2] - true_goals[(i-1)*2+1:i*2])
            end

            # Calculate total goal position errors
            method_goal_errors[trial, t-planning_horizon+1] = norm(method_goals - true_goals)
            baseline_goal_errors[trial, t-planning_horizon+1] = norm(baseline_goals - true_goals)
        end
    end

    # Save results
    open("monte_carlo_rh_results.txt", "w") do f
        write(f, "Method Goal Errors:\n")
        for trial in 1:num_trials
            write(f, "Trial $trial: ")
            write(f, string(round.(method_goal_errors[trial,:]; digits=4)), "\n")
        end
        write(f, "\nBaseline Goal Errors:\n")
        for trial in 1:num_trials
            write(f, "Trial $trial: ")
            write(f, string(round.(baseline_goal_errors[trial,:]; digits=4)), "\n")
        end
        write(f, "\nMethod Player Errors:\n")
        for trial in 1:num_trials
            write(f, "Trial $trial:\n")
            for player in 1:length(tracks)
                write(f, "  Player $player: ")
                write(f, string(round.(method_player_errors[trial,:,player]; digits=4)), "\n")
            end
        end
        write(f, "\nBaseline Player Errors:\n")
        for trial in 1:num_trials
            write(f, "Trial $trial:\n")
            for player in 1:length(tracks)
                write(f, "  Player $player: ")
                write(f, string(round.(baseline_player_errors[trial,:,player]; digits=4)), "\n")
            end
        end
    end

    method_mean = mean(method_goal_errors, dims=1)
    method_std = std(method_goal_errors, dims=1)
    baseline_mean = mean(baseline_goal_errors, dims=1)
    baseline_std = std(baseline_goal_errors, dims=1)

    println("\nResults Summary:")
    println("Time Step | Method Mean ± Std | Baseline Mean ± Std")
    println("------------------------------------------------")
    for t in 1:size(method_goal_errors, 2)
        println("$t | $(round(method_mean[t], digits=4)) ± $(round(method_std[t], digits=4)) | $(round(baseline_mean[t], digits=4)) ± $(round(baseline_std[t], digits=4))")
    end

    ExperimentGraphingUtils.plot_monte_carlo_goal_errors(method_goal_errors, baseline_goal_errors, method_player_errors, baseline_player_errors)

    return method_goal_errors, baseline_goal_errors, method_player_errors, baseline_player_errors
end

end