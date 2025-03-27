function solve_inverse_mcp_game(
    mcp_game,
    initial_state,
    τs_observed,
    initial_estimation,
    total_horizon;
    observation_model = identity,
    max_grad_steps = 150, lr = 1e-5, last_solution = nothing, discount_threshold = 1e-4,
)
    # Store all trajectories for animation
    all_trajectories = []
    
    function observe_trajectory(x)
        vcat([observation_model(state_t) for state_t in x.blocks]...)
    end
    """
    solve inverse game

    gradient steps using differentiable game solver on the observation likelihood loss
    """
    function likelihood_cost(τs_observed, context_state_estimation, initial_state)
        solution = solve_mcp_game(mcp_game, initial_state, 
            context_state_estimation; initial_guess = last_solution, total_horizon = total_horizon)
        # if solution.status != PATHSolver.MCP_Solved
        #     @info "Inner solve did not converge properly, re-initializing..."
        #     solution = solve_mcp_game(mcp_game, initial_state, 
        #         context_state_estimation; initial_guess = nothing, total_horizon = total_horizon)
        # end
        push!(solving_info, solution.info)
        last_solution = solution.status == PATHSolver.MCP_Solved ? (; primals = ForwardDiff.value.(solution.primals),
        variables = solution.variables, status = solution.status) : nothing
        τs_solution = reconstruct_solution(solution, mcp_game.game, total_horizon)
        ChainRulesCore.ignore() do
        push!(all_trajectories, ForwardDiff.value.(deepcopy(τs_solution)))
        end
        
        observed_τs_solution = observe_trajectory(τs_solution)
        
        if solution.status == PATHSolver.MCP_Solved
            infeasible_counter = 0
        else
            infeasible_counter += 1
        end
        norm_sqr(vcat(τs_observed...) - observed_τs_solution)
    end
    num_player = num_players(mcp_game.game)
    infeasible_counter = 0
    solving_info = []
    context_state_estimation = initial_estimation
    i_ = 0
    time_exec = 0
    grad_step_time = 0
    gradient = nothing
    for i in 1:max_grad_steps
        i_ = i
        
        # FORWARD diff
        # try
            grad_step_time = @elapsed gradient = Zygote.gradient(τs_observed, context_state_estimation, initial_state) do τs_observed, context_state_estimation, initial_state
                Zygote.forwarddiff([context_state_estimation; initial_state]; chunk_threshold = length(context_state_estimation) + length(initial_state)) do θ
                    context_state_estimation = BlockVector(θ[1:length(context_state_estimation)], blocksizes(context_state_estimation)[1])
                    initial_state = BlockVector(θ[(length(context_state_estimation) + 1):end], blocksizes(initial_state)[1])
                    likelihood_cost(τs_observed, context_state_estimation, initial_state)
                end
            end
        # catch e
        #     println(e)
        #     @info "Gradient step failed"
        #     return (; context_state_estimation, last_solution, i_, solving_info, time_exec, all_trajectories)
        # end
        time_exec += grad_step_time
        objective_grad = gradient[2]
        x0_grad = gradient[3]
        clamp!(objective_grad, -50, 50)
        clamp!(x0_grad, -10, 10)
        objective_update = lr * objective_grad
        x0_update = 1e-3 * x0_grad
        if norm(objective_update) < 1e-2 && norm(x0_update) < 1e-4
            @info "Inner iteration terminates at iteration: "*string(i)
            break
        elseif infeasible_counter >= 4
            @info "Inner iteration reached the maximal infeasible steps"
            break
        end
        context_state_estimation -= objective_update
        initial_state -= x0_update
    end
    (; context_state_estimation, last_solution, i_, solving_info, time_exec, all_trajectories)
end

# Add a new function to create an animation
function animate_optimization_progress(all_trajectories; fps=5)
    n_frames = length(all_trajectories)
    anim = Plots.@animate for i in 1:n_frames
        trajectory = all_trajectories[i]
        
        # Extract x and y coordinates from the trajectory
        # Adjust this based on your state representation
        xs = [state[1] for state in trajectory.blocks]
        ys = [state[2] for state in trajectory.blocks]
        
        Plots.plot(xs, ys, 
             label="Iteration $i",
             title="Trajectory Optimization Progress",
             xlabel="X Position",
             ylabel="Y Position",
             legend=:topleft)
        
        # Optionally plot the observed trajectory for comparison
        # plot!(observed_xs, observed_ys, label="Observed", line=:dash)
        
        # Add any additional plotting customization here
    end
    
    Plots.gif(anim, "trajectory_optimization.gif", fps=fps)
end

