using Infiltrator
function solve_myopic_inverse_game(
    mcp_game,
    observed_trajectory,
    observation_model,
    hidden_state_dim;
    initial_state = nothing,
    hidden_state_guess = nothing,
    max_grad_steps = 200,
    retries_on_divergence = 3,
    verbose = false,
    rng = Random.MersenneTwister(1),
    dynamics = nothing,
    lr = 1e-3,
    total_horizon = 25
)
    !verbose || println("solving ... ")
    initial_state = (isnothing(initial_state)) ? BlockVector(deepcopy(observed_trajectory[1]), collect(blocksizes(observed_trajectory[1], 1))) : initial_state
    !verbose || println("initial state: ", initial_state)
    hidden_state_0 = isnothing(hidden_state_guess) ?
        BlockVector(randn(rng, sum(hidden_state_dim)), collect(hidden_state_dim)) :
        BlockVector(hidden_state_guess, collect(hidden_state_dim))
    !verbose || println("hidden state: ", hidden_state_0)
    
    # warm_start_sol = 
    #     expand_warm_start(
    #         warm_start(
    #             observed_trajectory,
    #             initial_state,
    #             rh_horizon;
    #             num_players = num_players(mcp_game.game),
    #             observation_model = observation_model,
    #             partial_observation_state_size = Int64(size(observed_trajectory[1], 1) // num_players(mcp_game.game)),
    #             dynamics = dynamics),
    #         mcp_game)
    
    # if verbose 
    #     open("warm_start_sol.txt", "w") do io
    #         write(io, string(warm_start_sol))
    #     end
    #     !verbose || println("warm start sol: ", warm_start_sol.status)
    # end
    #TODO would be nice to check if warm start solution is feasible
    # for attempt in 1:retries_on_divergence
        # try
            context_state_estimation, last_solution, i_, solving_info, time_exec, all_trajectories = 
                solve_inverse_mcp_game(
                    mcp_game,
                    initial_state,
                    observed_trajectory,
                    hidden_state_0,
                    total_horizon;
                    observation_model = observation_model,         
                    max_grad_steps = max_grad_steps,
                    # last_solution = warm_start_sol,
                    lr = lr,
                )
            @info "inverse game took $(time_exec) seconds"
            animate_optimization_progress(all_trajectories)
            # verbose||println("solved, status: ", last_solution.status)
            # if solving_info[end].status == PATHSolver.MCP_Solved
                inv_sol = reconstruct_solution(solve_mcp_game(mcp_game, initial_state, context_state_estimation; verbose = false), mcp_game.game, mcp_game.horizon)
                inv_sol = vcat([observation_model(x) for x in inv_sol.blocks]...)
                sol_error = norm_sqr(inv_sol - vcat(observed_trajectory...))


                return (;
                sol_error = sol_error,
                recovered_params = context_state_estimation,
                recovered_trajectory = inv_sol,
                warm_start_trajectory = reconstruct_solution(warm_start_sol, mcp_game.game, mcp_game.horizon),
                solving_info = solving_info,
                time_exec = time_exec,
                )
            # end
        # catch e
        # end
    # end
end