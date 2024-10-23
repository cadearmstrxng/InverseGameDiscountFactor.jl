using Infiltrator

function solve_inverse_mcp_game(
    mcp_game::MCPGame,
    initial_state,
    τs_observed,
    initial_estimation,
    horizon;
    observation_model = (; σ = 0.0, expected_observation = identity),
    max_grad_steps = 150, lr = 1e-3, last_solution = nothing,
)
    function observe_trajectory(x)
        vcat([observation_model.expected_observation(state_t) for state_t in x.blocks]...)
    end
    """
    solve inverse game

    gradient steps using differentiable game solver on the observation likelihood loss
    """
    function likelihood_cost(τs_observed, context_state_estimation, initial_state)
        solution = solve_mcp_game(mcp_game, initial_state, 
            context_state_estimation; initial_guess = last_solution)
        if solution.status != PATHSolver.MCP_Solved
            @info "Inner solve did not converge properly, re-initializing..."
            solution = solve_mcp_game(mcp_game, initial_state, 
                context_state_estimation; initial_guess = nothing)
        end
        push!(solving_info, solution.info)
        last_solution = solution.status == PATHSolver.MCP_Solved ? (; primals = ForwardDiff.value.(solution.primals),
        variables = ForwardDiff.value.(solution.variables), status = solution.status) : nothing
        τs_solution = reconstruct_solution(solution, mcp_game.game, horizon)
        observed_τs_solution = observe_trajectory(τs_solution)
        
        if solution.status == PATHSolver.MCP_Solved
            infeasible_counter = 0
        else
            infeasible_counter += 1
        end
        norm_sqr(τs_observed - observed_τs_solution)
    end
    num_player = num_players(mcp_game.game)
    infeasible_counter = 0
    solving_info = []
    context_state_estimation = initial_estimation
    i_ = 0
    time_exec = 0
    for i in 1:max_grad_steps
        i_ = i
        # clip the estimation by the lower and upper bounds
        # for ii in 1:num_player
        #     context_state_estimation[Block(ii)] = clamp.(context_state_estimation[Block(ii)], [-0.2, 0], [0.2, 1])
        # end
        
        # FORWARD diff
        grad_step_time = @elapsed gradient = Zygote.gradient(τs_observed, context_state_estimation, initial_state) do τs_observed, context_state_estimation, initial_state
            Zygote.forwarddiff([context_state_estimation; initial_state]; chunk_threshold = length(context_state_estimation) + length(initial_state)) do θ
                context_state_estimation = BlockVector(θ[1:length(context_state_estimation)], blocksizes(context_state_estimation)[1])
                initial_state = BlockVector(θ[(length(context_state_estimation) + 1):end], blocksizes(initial_state)[1])
                likelihood_cost(τs_observed, context_state_estimation, initial_state)
            end
        end
        time_exec += grad_step_time
        objective_grad = gradient[2]
        x0_grad = gradient[3]
        clamp!(objective_grad, -50, 50)
        clamp!(x0_grad, -10, 10)
        objective_update = lr * objective_grad
        x0_update = 1e-3 * x0_grad
        if norm(objective_update) < 1e-4 && norm(x0_update) < 1e-4
            @info "Inner iteration terminates at iteration: "*string(i)
            break
        elseif infeasible_counter >= 4
            @info "Inner iteration reached the maximal infeasible steps"
            break
        end
        context_state_estimation -= objective_update
        initial_state -= x0_update
    end
    (; context_state_estimation, last_solution, i_, solving_info, time_exec)
end
#     (
#     mcp_game :: MCPGame,
#     game,
#     τ_observed,
#     x0;
#     observation_index = nothing,
#     dim_params = 3,
#     initial_guess = nothing,
#     prior_parmas = nothing,
#     horizon,
# )
#     (; dynamics) = game
#     if !isnothing(initial_guess)
#         z = initial_guess.variables
#     else # start with the observation
#         z = zeros(length(inverse_problem.lb))
#         control_block_dimensions =
#             [control_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
#         state_dimension = state_dim(dynamics)
#         dummy_strategy =
#             (x, t) -> BlockVector(zeros(sum(control_block_dimensions)), control_block_dimensions)
#         xs = rollout(dynamics, dummy_strategy, x0, horizon + 1).xs[2:end]
#         xs = reduce(vcat, xs)
#         z[(dim_params + 1):(dim_params + state_dimension * horizon)] = xs
#     end

#     @infiltrate

#     lb = inverse_problem.lb
#     ub = inverse_problem.ub

#     function F(n, z, f)
#         if isnothing(prior_parmas)
#             inverse_problem.fill_F!(f, z, x0, τ_observed)
#         else
#             inverse_problem.fill_F!(f, z, x0, τ_observed, prior_parmas)
#         end

#         Cint(0)
#     end

#     function J(n, nnz, z, col, len, row, data)
#         if isnothing(prior_parmas)
#             inverse_problem.fill_J(inverse_problem.fill_J.result_buffer, z, x0, τ_observed)
#         else
#             inverse_problem.fill_J(inverse_problem.fill_J.result_buffer, z, x0, τ_observed, prior_parmas)
#         end
#         ParametricMCPs._coo_from_sparse!(col, len, row, data, inverse_problem.fill_J.result_buffer)

#         Cint(0)
#     end

#     # count the non-zeros in J matrix
#     nnz = length(inverse_problem.fill_J.rows)

#     @infiltrate

#     status, variables, info = PATHSolver.solve_mcp(
#         F,
#         J,
#         lb,
#         ub,
#         z;
#         silent = true,
#         nnz,
#         cumulative_iteration_limit = 100_000,
#         use_basics = true,
#         use_start = true,
#         jacobian_structure_constant = true,
#         jacobian_data_contiguous = true,
#         jacobian_linear_elements = inverse_problem.fill_J.constant_entries,
#     )

#     if status != PATHSolver.MCP_Solved
#         @warn "MCP not cleanly solved. Final solver status is $(status)."
#     end

#     (; variables, status, info)
# end
