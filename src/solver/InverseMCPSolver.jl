function solve_inverse_mcp_game(
    mcp_game,
    initial_state,
    τs_observed,
    initial_estimation,
    total_horizon;
    observation_model = identity,
    max_grad_steps = 150, 
    lr = 1e-5, 
    last_solution = nothing, 
    discount_threshold = 1e-4,
    use_adaptive_lr = true,
    lr_reduction_factor = 0.5,
    lr_increase_factor = 1.1,
    min_lr = 1e-7,
    max_consecutive_success = 3
)
    # Store all trajectories for animation
    all_trajectories = []
    last_solution_ref = Ref{Any}(last_solution)
    
    function observe_trajectory(x)
        vcat([observation_model(state_t) for state_t in x.blocks]...)
    end
    """
    solve inverse game

    gradient steps using differentiable game solver on the observation likelihood loss
    """
    function likelihood_cost(τs_observed, context_state_estimation, initial_state)
        solution = solve_mcp_game(mcp_game, initial_state, 
            context_state_estimation; initial_guess = last_solution_ref[], total_horizon = total_horizon)
        # if solution.status != PATHSolver.MCP_Solved
        #     @info "Inner solve did not converge properly, re-initializing..."
        #     solution = solve_mcp_game(mcp_game, initial_state, 
        #         context_state_estimation; initial_guess = nothing, total_horizon = total_horizon)
        # end
        
        τs_solution = reconstruct_solution(solution, mcp_game.game, total_horizon)
        
        ChainRulesCore.ignore_derivatives() do
            push!(solving_info, solution.info)
            last_solution_ref[] = (; primals = ForwardDiff.value.(solution.primals),
                variables = ForwardDiff.value.(solution.variables), status = solution.status)
            push!(all_trajectories, ForwardDiff.value.(deepcopy(τs_solution)))
            # println("forward game success ratio: ", solution.success_ratio)
        
            was_successful[] = solution.success_ratio >= 0.9
        
            if was_successful[]
                last_known_good_context_state[] = ForwardDiff.value.(deepcopy(context_state_estimation))
                infeasible_counter[] = 0
                if use_adaptive_lr
                    consecutive_success_counter[] += 1
                end
            else
                infeasible_counter[] += 1
                if use_adaptive_lr
                    consecutive_success_counter[] = 0
                end
            end
        end

        observed_τs_solution = observe_trajectory(τs_solution)
        
        norm_sqr(vcat(τs_observed...) - observed_τs_solution)
    end
    num_player = num_players(mcp_game.game)
    infeasible_counter = Ref(0)
    last_known_good_context_state = Ref(initial_estimation)
    consecutive_success_counter = Ref(0)
    was_successful = Ref(false)
    current_lr = lr
    solving_info = []
    context_state_estimation = initial_estimation
    context_states = [context_state_estimation]
    i_ = 0
    time_exec = 0
    grad_step_time = 0
    gradient = nothing
    len_cse = length(context_state_estimation)
    bs_cse = [Int64(len_cse // num_player) for _ in 1:num_player]
    bs_is = blocksizes(initial_state)[1]
    for i in 1:max_grad_steps
        i_ = i
        
        grad_step_time = @elapsed gradient = Zygote.gradient(τs_observed, context_state_estimation, initial_state[1:end]) do τs_observed, context_state_estimation, initial_state
            Zygote.forwarddiff([context_state_estimation; initial_state]; chunk_threshold = len_cse + length(initial_state)) do θ
                context_state_estimation_dual = BlockVector(@view(θ[1:len_cse]), bs_cse)
                initial_state_dual = BlockVector(@view(θ[(len_cse + 1):end]), bs_is)
                likelihood_cost(τs_observed, context_state_estimation_dual, initial_state_dual)
            end
        end
        time_exec += grad_step_time
        
        objective_grad = gradient[2]
        x0_grad = gradient[3]

        if use_adaptive_lr
            
            if !was_successful[]
                current_lr = max(current_lr * lr_reduction_factor, min_lr)
            elseif consecutive_success_counter[] >= max_consecutive_success
                current_lr = min(current_lr * lr_increase_factor, lr) # don't exceed original lr
                consecutive_success_counter[] = 0 # Reset after increasing LR
            end
            
            if current_lr <= min_lr
                @warn "Learning rate collapsed. Stopping optimization."
                break
            end
        end

        clamp!(objective_grad, -50, 50)
        clamp!(x0_grad, -10, 10)
        objective_update = current_lr * objective_grad
        x0_update = current_lr * x0_grad
        
        if (norm(objective_grad) / norm(context_state_estimation) < 1e-4 && norm(x0_grad) / norm(initial_state) < 1e-4)
            @info "Inner iteration terminates at iteration: "*string(i)
            break
        elseif infeasible_counter[] >= 4
            @info "Inner iteration reached the maximal infeasible steps"
            return (; last_known_good_context_state = last_known_good_context_state[], last_solution = last_solution_ref[], i_, solving_info, time_exec, all_trajectories, context_states)
            break
        end
        context_state_estimation -= objective_update
        initial_state -= x0_update
    end
    context_state_estimation = ForwardDiff.value.(context_state_estimation)
    (; context_state_estimation, last_solution = last_solution_ref[], i_, solving_info, time_exec, all_trajectories, context_states)
end

# Add a new function to create an animation
# function animate_optimization_progress(all_trajectories, mcp_game; fps=1//3)
#     n_frames = length(all_trajectories)
#     n = num_players(mcp_game.game)
#     player_state_dimension = convert(Int64, state_dim(mcp_game.game.dynamics)/n)
#     horizon = mcp_game.horizon
#     anim = Plots.@animate for i in 1:n_frames
#         trajectory = all_trajectories[i]
        
#         # Create a single plot for all players
#         p = Plots.plot(title="Trajectory Optimization Progress (Frame $i/$n_frames)",
#                       xlabel="X Position",
#                       ylabel="Y Position",
#                       legend=:topleft)
        
#         # Add each player's trajectory to the same plot
#         for j in 1:n
#             xs = [state[(j-1) * player_state_dimension + 1] for state in trajectory.blocks[1:horizon-1]]
#             ys = [state[(j-1) * player_state_dimension + 2] for state in trajectory.blocks[1:horizon-1]]
#             Plots.plot!(p, xs, ys, label="Player $j")
#         end
#     end
    
#     Plots.gif(anim, "trajectory_optimization.gif", fps=fps)
#     display(anim)
# end

