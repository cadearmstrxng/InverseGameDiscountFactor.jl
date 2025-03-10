module MonteCarloSim

using Random
using BlockArrays: Block, mortar, BlockVector, blocksizes
using LinearAlgebra: norm_sqr
using TrajectoryGamesBase: num_players, state_dim
using TrajectoryGamesExamples: BicycleDynamics
using Statistics: mean, std
using HypothesisTests: OneSampleTTest, pvalue

include("../src/InverseGameDiscountFactor.jl")
include("GameUtils.jl")
include("graphing/ExperimentGraphingUtils.jl")
include("In-D/InD.jl")

function run_full_state_monte_carlo(;
    frames = [26158, 26320],
    tracks = [201, 205, 207, 208],
    downsample_rate = 6,
    rng = Random.MersenneTwister(1),
    num_trials = 50,
    σs = [0.002*i for i in 0:50],
    # σs = [0.00],
    verbose = true,
    store_all = false
)
    # Set randomness
    Random.seed!(rng)

    # Clear previous error files
    open("experiments/In-D/mc_study_new_traj_errors.txt", "w") do f
        # Opening in write mode clears the file
    end
    open("experiments/In-D/mc_study_baseline_traj_errors.txt", "w") do f
        # Opening in write mode clears the file
    end

    # Get real trajectory data
    InD_observations = GameUtils.pull_trajectory("07";
        track = tracks, 
        downsample_rate = downsample_rate, 
        all = false, 
        frames = frames
    )

    trk_201_lane_center(x) = 0.0  # Placeholder
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
        true;
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
    errors = Array{Float64}(undef, num_trials)
    baseline_errors = Array{Float64}(undef, num_trials)
    
    # observed_trajectories = []
    # recovered_trajectories = []
    # recovered_params = []

    for σ_idx in eachindex(σs)
        σ = σs[σ_idx]
        errors .= -1.0
        baseline_errors .= -1.0
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
            errors[trial_idx] = norm_sqr(vcat(InD_observations...) - method_sol.recovered_trajectory) / length(InD_observations)
            baseline_errors[trial_idx] = norm_sqr(vcat(InD_observations...) - baseline_sol.recovered_trajectory) / length(InD_observations)
            
        end
        open("experiments/In-D/mc_study_new_traj_errors.txt", "a") do f
            write(f, "m"*string(σ)*" "*join(string.(round.(errors, digits=4)), " ")*"\n")
        end
        open("experiments/In-D/mc_study_baseline_traj_errors.txt", "a") do f
            write(f, "b"*string(σ)*" "*join(string.(round.(baseline_errors, digits=4)), " ")*"\n")
        end
    end
    println("mc study done")
    
    # Generate summary statistics
    generate_monte_carlo_summary(σs, num_trials)
end

"""
    generate_monte_carlo_summary(σs, num_trials)

Generate summary statistics from Monte Carlo error files.
Reads the error files and computes statistical metrics including:
- Average improvement
- Percentage improvement
- Standard deviations
- Statistical significance tests

Results are written to a summary file.
"""
function generate_monte_carlo_summary(σs, num_trials)
    # Create or clear the summary file
    open("experiments/In-D/mc_study_summary.txt", "w") do f
        write(f, "# Monte Carlo Study Summary\n\n")
    end
    
    for σ in σs
        # Read the error data from files
        new_errors = Float64[]
        baseline_errors = Float64[]
        
        # Read new method errors
        open("experiments/In-D/mc_study_new_traj_errors.txt", "r") do f
            for line in eachline(f)
                if startswith(line, "m"*string(σ)*" ")
                    error_strings = split(line[length("m"*string(σ)*" "):end])
                    append!(new_errors, parse.(Float64, error_strings))
                end
            end
        end
        
        # Read baseline errors
        open("experiments/In-D/mc_study_baseline_traj_errors.txt", "r") do f
            for line in eachline(f)
                if startswith(line, "b"*string(σ)*" ")
                    error_strings = split(line[length("b"*string(σ)*" "):end])
                    append!(baseline_errors, parse.(Float64, error_strings))
                end
            end
        end
        
        # Check for invalid error values (-1.0)
        invalid_new = count(x -> x == -1.0, new_errors)
        invalid_baseline = count(x -> x == -1.0, baseline_errors)
        
        if invalid_new > 0
            @warn "Found $invalid_new invalid values (-1.0) in new method errors for σ=$σ"
        end
        
        if invalid_baseline > 0
            @warn "Found $invalid_baseline invalid values (-1.0) in baseline errors for σ=$σ"
        end
        
        # Filter out invalid values for statistics calculation
        filtered_new_errors = filter(x -> x != -1.0, new_errors)
        filtered_baseline_errors = filter(x -> x != -1.0, baseline_errors)
        
        # Ensure we have enough valid data points
        if length(filtered_new_errors) == 0 || length(filtered_baseline_errors) == 0
            @error "No valid error data for σ=$σ after filtering invalid values"
            continue
        end
        
        # Also check if we have the right number of data points
        if length(new_errors) != num_trials || length(baseline_errors) != num_trials
            @warn "Expected $num_trials trials but found $(length(new_errors)) new method errors and $(length(baseline_errors)) baseline errors for σ=$σ"
        end
        
        # Calculate statistics using filtered data
        open("experiments/In-D/mc_study_summary.txt", "a") do f
            write(f, "## Noise level σ = $σ\n\n")
            
            if invalid_new > 0 || invalid_baseline > 0
                write(f, "⚠️ Warning: Found $(invalid_new + invalid_baseline) invalid data points (-1.0) that were excluded from analysis\n\n")
            end
            
            average_improvement = mean(filtered_baseline_errors) - mean(filtered_new_errors)
            write(f, "average_improvement @ $σ $(round(average_improvement, digits=4))\n")

            average_percentage_improvement = average_improvement / mean(filtered_baseline_errors) * 100
            write(f, "average_percentage_improvement @ $σ $(round(average_percentage_improvement, digits=4))\n")
            
            baseline_std = std(filtered_baseline_errors)
            new_std = std(filtered_new_errors)
            write(f, "baseline_std @ $σ $(round(baseline_std, digits=4))\n")
            write(f, "new_std @ $σ $(round(new_std, digits=4))\n")
            
            # Perform paired t-test to check statistical significance only if we have paired data
            valid_pairs = [(b, n) for (b, n) in zip(baseline_errors, new_errors) if b != -1.0 && n != -1.0]
            
            if length(valid_pairs) > 1
                # For paired t-test, we test if the difference (baseline - new) is significantly > 0
                error_differences = [pair[1] - pair[2] for pair in valid_pairs]
                t_test = OneSampleTTest(error_differences)
                p_val = pvalue(t_test, tail=:right)  # right-tailed test for baseline > new
                
                write(f, "p_value @ $σ $(round(p_val, digits=6))\n")
                write(f, "significant @ $σ $(p_val < 0.05)\n")
                
                # Calculate likelihood as a percentage
                significance_likelihood = (1 - p_val) * 100
                write(f, "likelihood_of_improvement @ $σ $(round(significance_likelihood, digits=2))%\n\n")
            else
                write(f, "p_value @ $σ N/A (insufficient valid paired data for statistical testing)\n\n")
            end
        end
    end
    println("Summary statistics generated")
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