function solve_myopic_inverse_game(
    mcp_game,
    observed_trajectory,
    observation_model,
    hidden_state_dim;
    hidden_state_guess = nothing,
    max_grad_steps = 200,
    retries_on_divergence = 3,
    verbose = false,
    rng = Random.MersenneTwister(1)
)
    initial_state = deepcopy(observed_trajectory[Block(1)])
    hidden_state_0 = isnothing(hidden_state_guess) ?
        BlockVector(randn(rng, sum(hidden_state_dim)), hidden_state_dim) :
        BlockVector(hidden_state_guess, hidden_state_dim)
    
    warm_start_sol = 
        expand_warm_start(
            warm_start(
                observed_trajectory,
                initial_state,
                mcp_game.horizon;
                observation_model = observation_model,
                partial_observation_state_size = int64(size(observed_trajectory[Block(1)])[1] // num_players(mcp_game.game))),
                mcp_game)
    #TODO would be nice to check if warm start solution is feasible
    for _ in 1:retries_on_divergence
        try
            context_state_estimation, last_solution, i_, solving_info, time_exec = 
                solve_inverse_mcp_game(
                    mcp_game,
                    initial_state,
                    observed_trajectory,
                    hidden_state_0,
                    mcp_game.horizon;
                    observation_model = observation_model,         
                    max_grad_steps = max_grad_steps,
                    last_solution = warm_start_sol
                )
            if solving_info[end].status == PATHSolver.MCP_Solved
                inv_sol = solve_mcp_game(mcp_game, initial_state, context_state_estimation; verbose = false)
                sol_error = norm_sqr(reconstruct_solution(for_sol, mcp_game, horizon) 
                    - reconstruct_solution(inv_sol, mcp_game, horizon))


                return (;
                sol_error = sol_error,
                recovered_params = context_state_estimation,
                recovered_trajectory = reconstruct_solution(inv_sol, mcp_game, mcp_game.horizon),
                solving_info = solving_info,
                time_exec = time_exec,
                )
            end
        catch e
        end
    end
end