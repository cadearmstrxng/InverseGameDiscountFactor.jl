module MonteCarloSim

using Random
using BlockArrays: Block, mortar, BlockVector, blocksizes
using LinearAlgebra: norm_sqr
using TrajectoryGamesBase: num_players, state_dim
using TrajectoryGamesExamples: BicycleDynamics
using Statistics: mean

include("../src/InverseGameDiscountFactor.jl")
include("GameUtils.jl")
include("graphing/ExperimentGraphingUtils.jl")
include("In-D/InD.jl")

function run_full_state_monte_carlo(;
    frames = [26158, 26320],
    tracks = [201, 205, 207, 208],
    downsample_rate = 9,
    rng = Random.MersenneTwister(1),
    num_trials = 50,
    # σs = [0.01*i for i in 0:2:10],
    σs = [0.00],
    verbose = true,
    store_all = false
)
    # Set randomness
    Random.seed!(rng)

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )
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
        true;
        initial_state = InD_observations[1],
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] for i in 1:length(tracks)]...]),
        horizon = length(frames[1]:downsample_rate:frames[2]),
        n = length(tracks),
        dt = 0.04*downsample_rate,
        myopic=true,
        verbose = false,
        dynamics = dynamics,
        lane_centers = lane_centers
    )

    init_baseline = GameUtils.init_bicycle_test_game(
        true;
        initial_state = init.initial_state,
        game_params = mortar([
            [[InD_observations[end][Block(i)][1:2]..., 1.0, 1.0, 1.0, 1.0, 1.0] for i in 1:length(tracks)]...]),
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
    
    # Track errors
    errors = Array{Float64}(undef, length(σs), num_trials)
    # parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    baseline_errors = Array{Float64}(undef, length(σs), num_trials)
    # baseline_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    
    observed_trajectories = []
    recovered_trajectories = []
    recovered_params = []

    for σ_idx in eachindex(σs)
        σ = σs[σ_idx]
        for trial_idx in 1:num_trials
            verbose && println("std: ", σ, " trial: ", trial_idx)

            # Add noise to real observations
            noisy_observations = map(InD_observations) do obs
                BlockVector(init.observation_model(obs, σ=σ),
                    [Int64(state_dim(init.game_structure.game.dynamics) ÷ num_players(init.game_structure.game)) 
                    for _ in 1:num_players(init.game_structure.game)])
            end

            # Solve inverse game with both methods
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

            # Record errors for both methods
            errors[σ_idx, trial_idx] = norm_sqr(vcat(InD_observations...) - method_sol.recovered_trajectory) / length(InD_observations)
            parameter_errors[σ_idx, trial_idx] = norm_sqr(init.game_parameters - method_sol.recovered_params)
            baseline_errors[σ_idx, trial_idx] = norm_sqr(vcat(InD_observations...) - baseline_sol.recovered_trajectory) / length(InD_observations)
            baseline_parameter_errors[σ_idx, trial_idx] = norm_sqr(init_baseline.game_parameters - baseline_sol.recovered_params)
            
            if store_all
                push!(observed_trajectories, noisy_observations)
                push!(recovered_trajectories, method_sol.recovered_trajectory)
                push!(recovered_params, method_sol.recovered_params)
            end
        end
    end
    println("mc study done")
    open("experiments/In-D/mc_study_new_traj_errors.txt", "a") do f
        for i in 1:length(σs)
            write(f, string(round.(errors[i, :], digits=4), "\n"))
        end
    end
    open("experiments/In-D/mc_study_baseline_traj_errors.txt", "a") do f
        for i in 1:length(σs)
            write(f, string(round.(baseline_errors[i, :], digits=4), "\n"))
        end
    end
    # open("experiments/In-D/mc_study_new_param_errors.txt", "a") do f
    #     for i in 1:length(σs)
    #         write(f, string(round.(parameter_errors[i, :], digits=4), "\n"))
    #     end
    # end
    # open("experiments/In-D/mc_study_baseline_param_errors.txt", "a") do f
    #     for i in 1:length(σs)
    #         write(f, string(round.(baseline_parameter_errors[i, :], digits=4), "\n"))
    #     end
    # end

    # Graph results
    ExperimentGraphingUtils.graph_metrics(
        baseline_errors,
        errors,
        baseline_parameter_errors,
        parameter_errors,
        σs;
        observation_mode="InD_full_state",
        pre_prefix = "experiments/In-D/"
    )

    return (;
        errors = errors,
        parameter_errors = parameter_errors,
        observed_trajectories = observed_trajectories,
        recovered_trajectories = recovered_trajectories,
        recovered_params = recovered_params
    )
end

# include("baseline_old/Baseline.jl")
# # include("crosswalk_sim/Crosswalk.jl")

# function run_monte_carlo_sims(;
#     initial_state = mortar([
#         [0, 2, 0.1, -0.2],
#         [2.5, 2, 0.0, 0.0],
#     ]),
#     hidden_params = mortar([[2, 0, 0.6], [0, 0, 0.6]]),
#     rng = Random.MersenneTwister(1),
#     num_trials = 40,
#     σs = [0.01*i for i in 0:10],
#     verbose = false,
#     store_all = false
# )
#     # Set randomoness
#     Random.seed!(rng)

#     # Full state observability
#     fsb_game = init_baseline_crosswalk_game(true)
#     fsm_game = init_myopic_crosswalk_game(true)

#     # Partial state observability
#     psb_game = init_baseline_crosswalk_game(false)
#     psm_game = init_myopic_crosswalk_game(false)

#     fsb_mcp_game = MCPCoupledOptimizationSolver(
#         fsb_game.game_structure.game,
#         fsb_game.horizon,
#         blocksizes(fsb_game.game_parameters, 1)
#         ).game
#     fsm_mcp_game = MCPCoupledOptimizationSolver(
#         fsm_game.game_structure.game,
#         fsm_game.horizon,
#         blocksizes(fsm_game.game_parameters, 1)
#         ).game
#     psb_mcp_game = MCPCoupledOptimizationSolver(
#         psb_game.game_structure.game,
#         psb_game.horizon,
#         blocksizes(psb_game.game_parameters, 1)
#         ).game
#     psm_mcp_game = MCPCoupledOptimizationSolver(
#         psm_game.game_structure.game,
#         psm_game.horizon,
#         blocksizes(psm_game.game_parameters, 1)
#         ).game

#     #TODO forward game is always fully observable?
#     forward_solution = solve_mcp_game(fsm_mcp_game, initial_state, hidden_params; verbose = false)
#     flattened_forward_solution = reconstruct_solution(forward_solution)


#     # track errors
#     fsb_errors = Array{Float64}(undef, length(σs), num_trials)
#     fsm_errors = Array{Float64}(undef, length(σs), num_trials)
#     psb_errors = Array{Float64}(undef, length(σs), num_trials)
#     psm_errors = Array{Float64}(undef, length(σs), num_trials)

#     fsb_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
#     fsm_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
#     psb_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
#     psm_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    
#     fs_observed_trajectories = []
#     ps_observed_trajectories = []
    
#     fsb_recovered_traj = []
#     fsm_recovered_traj = []
#     psb_recovered_traj = []
#     psm_recovered_traj = []

#     fsb_recovered_params = []
#     fsm_recovered_params = []
#     psb_recovered_params = []
#     psm_recovered_params = []

#     for σ_idx in eachindex(σs)
#         σ = σs[σ_idx]
#         for trial_idx in 1:num_trials
#             verbose || println("std: ", σ, " trial: ", i)

#             fs_observed_trajectory = BlockVector( 
#                 fsb_game.observation_model(flattened_forward_solution, σ=σ),
#                 [Int64(state_dim(fsb_game)//num_players(fsb_game)) for _ in 1:fsb_game.horizon])
#             ps_observed_trajectory = BlockVector( 
#                 psb_game.observation_model(flattened_forward_solution, σ=σ),
#                 [Int64(state_dim(fsb_game)//num_players(fsb_game)) for _ in 1:fsb_game.horizon])

#             # TODO use solver
#             fsb_results = nothing
#             fsm_results = nothing
#             psb_results = nothing
#             psm_results = nothing

#             fsb_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(fsb_results.recovered_trajectory))
#             fsm_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(fsm_results.recovered_trajectory))
#             psb_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(psb_results.recovered_trajectory))
#             psm_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(psm_results.recovered_trajectory))

#             fsb_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
#             fsm_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
#             psb_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
#             psm_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
            
#             if store_all
#                 push!(fs_observed_trajectories, fs_observed_trajectory)
#                 push!(ps_observed_trajectories, ps_observed_trajectory)
                
#                 push!(fsb_recovered_traj, fsb_results.recovered_trajectory)
#                 push!(fsm_recovered_traj, fsm_results.recovered_trajectory)
#                 push!(psb_recovered_traj, psb_results.recovered_trajectory)
#                 push!(psm_recovered_traj, psm_results.recovered_trajectory)

#                 push!(fsb_recovered_params, fsb_results.recovered_params)
#                 push!(fsm_recovered_params, fsm_results.recovered_params)
#                 push!(psb_recovered_params, psb_results.recovered_params)
#                 push!(psm_recovered_params, psm_results.recovered_params)
#             end
#         end
#     end

#     graph_metrics(
#         fsb_errors,
#         fsm_errors,
#         fsb_parameter_errors,
#         fsm_parameter_errors,
#         σs;
#         observation_mode="full state"
#     )

#     graph_metrics(
#         psb_errors,
#         psm_errors,
#         psb_parameter_errors,
#         psm_parameter_errors,
#         σs;
#         observation_mode="partial state"
#     )
# end


end