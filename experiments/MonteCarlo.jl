module MonteCarloSim

using Random

include("baseline_old/Baseline.jl")
include("crosswalk_sim/Crosswalk.jl")

function run_monte_carlo_sims(;
    initial_state = mortar([
        [0, 2, 0.1, -0.2],
        [2.5, 2, 0.0, 0.0],
    ]),
    hidden_params = mortar([[2, 0, 0.6], [0, 0, 0.6]]),
    rng = Random.MersenneTwister(1),
    num_trials = 40,
    σs = [0.01*i for i in 0:10],
    verbose = false,
    store_all = false
)
    # Set randomoness
    Random.seed!(rng)

    # Full state observability
    fsb_game = init_baseline_crosswalk_game(true)
    fsm_game = init_myopic_crosswalk_game(true)

    # Partial state observability
    psb_game = init_baseline_crosswalk_game(false)
    psm_game = init_myopic_crosswalk_game(false)

    fsb_mcp_game = MCPCoupledOptimizationSolver(
        fsb_game.game_structure.game,
        fsb_game.horizon,
        blocksizes(fsb_game.game_parameters, 1)
        ).game
    fsm_mcp_game = MCPCoupledOptimizationSolver(
        fsm_game.game_structure.game,
        fsm_game.horizon,
        blocksizes(fsm_game.game_parameters, 1)
        ).game
    psb_mcp_game = MCPCoupledOptimizationSolver(
        psb_game.game_structure.game,
        psb_game.horizon,
        blocksizes(psb_game.game_parameters, 1)
        ).game
    psm_mcp_game = MCPCoupledOptimizationSolver(
        psm_game.game_structure.game,
        psm_game.horizon,
        blocksizes(psm_game.game_parameters, 1)
        ).game

    #TODO forward game is always fully observable?
    forward_solution = solve_mcp_game(fsm_mcp_game, initial_state, hidden_params; verbose = false)
    flattened_forward_solution = reconstruct_solution(forward_solution)


    # track errors
    fsb_errors = Array{Float64}(undef, length(σs), num_trials)
    fsm_errors = Array{Float64}(undef, length(σs), num_trials)
    psb_errors = Array{Float64}(undef, length(σs), num_trials)
    psm_errors = Array{Float64}(undef, length(σs), num_trials)

    fsb_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    fsm_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    psb_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    psm_parameter_errors = Array{Float64}(undef, length(σs), num_trials)
    
    fs_observed_trajectories = []
    ps_observed_trajectories = []
    
    fsb_recovered_traj = []
    fsm_recovered_traj = []
    psb_recovered_traj = []
    psm_recovered_traj = []

    fsb_recovered_params = []
    fsm_recovered_params = []
    psb_recovered_params = []
    psm_recovered_params = []

    for σ_idx in eachindex(σs)
        σ = σs[σ_idx]
        for trial_idx in 1:num_trials
            verbose || println("std: ", σ, " trial: ", i)

            fs_observed_trajectory = BlockVector( 
                fsb_game.observation_model(flattened_forward_solution, σ=σ),
                [Int64(state_dim(fsb_game)//num_players(fsb_game)) for _ in 1:fsb_game.horizon])
            ps_observed_trajectory = BlockVector( 
                psb_game.observation_model(flattened_forward_solution, σ=σ),
                [Int64(state_dim(fsb_game)//num_players(fsb_game)) for _ in 1:fsb_game.horizon])

            # TODO use solver
            fsb_results = nothing
            fsm_results = nothing
            psb_results = nothing
            psm_results = nothing

            fsb_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(fsb_results.recovered_trajectory))
            fsm_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(fsm_results.recovered_trajectory))
            psb_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(psb_results.recovered_trajectory))
            psm_errors[σ_idx, trial_idx] = norm_sqr(flattened_forward_solution - reconstruct_solution(psm_results.recovered_trajectory))

            fsb_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
            fsm_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
            psb_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
            psm_parameter_errors[σ_idx, trial_idx] = norm_sqr(hidden_params - fsb_results.recovered_params)
            
            if store_all
                push!(fs_observed_trajectories, fs_observed_trajectory)
                push!(ps_observed_trajectories, ps_observed_trajectory)
                
                push!(fsb_recovered_traj, fsb_results.recovered_trajectory)
                push!(fsm_recovered_traj, fsm_results.recovered_trajectory)
                push!(psb_recovered_traj, psb_results.recovered_trajectory)
                push!(psm_recovered_traj, psm_results.recovered_trajectory)

                push!(fsb_recovered_params, fsb_results.recovered_params)
                push!(fsm_recovered_params, fsm_results.recovered_params)
                push!(psb_recovered_params, psb_results.recovered_params)
                push!(psm_recovered_params, psm_results.recovered_params)
            end
        end
    end

    graph_metrics(
        fsb_errors,
        fsm_errors,
        fsb_parameter_errors,
        fsm_parameter_errors,
        σs;
        observation_mode="full state"
    )

    graph_metrics(
        psb_errors,
        psm_errors,
        psb_parameter_errors,
        psm_parameter_errors,
        σs;
        observation_mode="partial state"
    )
end


end