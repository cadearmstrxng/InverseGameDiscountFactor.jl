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
    # use_warm_start=false,
    regularization = 0.0
)
    !verbose || println("solving ... ")
    initial_state = (isnothing(initial_state)) ? BlockVector(deepcopy(observed_trajectory[1]), collect(blocksizes(observed_trajectory[1], 1))) : initial_state
    !verbose || println("initial state: ", initial_state)
    hidden_state_0 = isnothing(hidden_state_guess) ?
        BlockVector(randn(rng, sum(hidden_state_dim)), collect(hidden_state_dim)) :
        BlockVector(hidden_state_guess, collect(hidden_state_dim))
    !verbose || println("hidden state: ", hidden_state_0)
    
    # if use_warm_start
    # warm_start_sol = 
    #     expand_warm_start(
    #         warm_start(
    #             observed_trajectory,
    #             initial_state,
    #             mcp_game.horizon;
    #             num_players = num_players(mcp_game.game),
    #             observation_model = observation_model,
    #             partial_observation_state_size = Int64(size(observed_trajectory[1], 1) // num_players(mcp_game.game)),
    #             dynamics = dynamics),
    #         mcp_game)
    # else
    #     warm_start_sol = nothing
    # end
    
    # if verbose && use_warm_start
    #     open("warm_start_sol.txt", "w") do io
    #         write(io, string(warm_start_sol))
    #     end
    #     !verbose || println("warm start sol: ", warm_start_sol.status)
    # end
    #TODO would be nice to check if warm start solution is feasible
    # for attempt in 1:retries_on_divergence
        # try
            context_state_estimation, last_solution, i_, solving_info, time_exec, solving_status = 
                solve_inverse_mcp_game(
                    mcp_game,
                    initial_state,
                    observed_trajectory,
                    hidden_state_0,
                    mcp_game.horizon;
                    observation_model = observation_model,         
                    max_grad_steps = max_grad_steps,
                    # last_solution = warm_start_sol,
                    lr = lr,
                    regularization = regularization
                )
            # verbose||println("solved, status: ", last_solution.status)
            # if solving_info[end].status == PATHSolver.MCP_Solved
                # inv_sol1 = reconstruct_solution(solve_mcp_game(mcp_game, initial_state, context_state_estimation; verbose = false), mcp_game.game, mcp_game.horizon)
                # inv_sol = vcat([observation_model(x) for x in inv_sol1.blocks]...)
                # sol_error = norm_sqr(inv_sol - vcat(observed_trajectory...))


                return (;
                # sol_error = sol_error,
                recovered_params = context_state_estimation,
                # recovered_trajectory = inv_sol1,
                # # warm_start_trajectory = use_warm_start ? reconstruct_solution(warm_start_sol, mcp_game.game, mcp_game.horizon) : nothing,
                # solving_info = solving_info,
                # time_exec = time_exec,
                # solving_status = solving_status
                )
            # end
        # catch e
        # end
    # end
end