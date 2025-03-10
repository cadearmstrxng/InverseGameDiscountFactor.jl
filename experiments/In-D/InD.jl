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
    trk_207_lane_center(x) = -6.535465682649165e-04*x^6 + 
                            -0.069559792458210*x^5 + 
                            -3.033950160533982*x^4 + 
                            -69.369975733866840*x^3 + 
                            -8.760325006936075e+02*x^2 + 
                            -5.782944928944775e+03*x + 
                            -1.547509969706588e+04

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
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for i in 1:length(tracks)]...]),
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
    InD_observations = full_state ? InD_observations : [observation[1:2] for observation in InD_observations]
    trk_201_lane_center(x) = 0.0  # Placeholder if no coefficients provided
    trk_205_lane_center(x) = 0.0  # Placeholder if no coefficients provided

    # 6th degree polynomial for track 207
    trk_207_lane_center(x) = -6.535465682649165e-04*x^6 + 
                            -0.069559792458210*x^5 + 
                            -3.033950160533982*x^4 + 
                            -69.369975733866840*x^3 + 
                            -8.760325006936075e+02*x^2 + 
                            -5.782944928944775e+03*x + 
                            -1.547509969706588e+04

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
        BlockVector(init.observation_model(obs, σ=σ),
            [Int64(state_dim(init.game_structure.game.dynamics) ÷ num_players(init.game_structure.game)) 
            for _ in 1:num_players(init.game_structure.game)])
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


    ExperimentGraphingUtils.graph_trajectories(
        "Observed v. Recovered v. Baseline",
        [InD_observations, method_sol.recovered_trajectory, baseline_sol.recovered_trajectory],
        init.game_structure,
        init.horizon;
        colors = [
            [(:red, 0.2), (:blue, 0.2), (:green, 0.2), (:purple, 0.2)],
            [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
            [(:red, 0.5 ), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)]
        ],
        constraints = init.environment === nothing ? nothing : get_constraints(init.environment)
    )

    ExperimentGraphingUtils.graph_trajectories(
        "Recovered v. Baseline",
        [InD_observations, method_sol.recovered_trajectory, baseline_sol.recovered_trajectory],
        init.game_structure,
        init.horizon;
        colors = [
            [(:red, 0.0), (:blue, 0.0), (:green, 0.0), (:purple, 0.0)],
            [(:red, 1.0), (:blue, 1.0), (:green, 1.0), (:purple, 1.0)],
            [(:red, 0.5 ), (:blue, 0.5), (:green, 0.5), (:purple, 0.5)]
        ],
        constraints = init.environment === nothing ? nothing : get_constraints(init.environment)
    )
        
        
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

end