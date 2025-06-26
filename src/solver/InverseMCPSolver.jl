function solve_inverse_mcp_game(
    mcp_game,
    initial_state,
    τs_observed,
    initial_estimation,
    horizon;
    observation_model = identity,
    max_grad_steps = 150, lr = 1, last_solution = nothing, discount_threshold = 1e-4, # lr usually 1e-3
    regularization = 0.0
    )
    function observe_trajectory(x)
        vcat([observation_model(state_t[1:end]) for state_t in x]...)
    end
    
    function loss(context_state_estimation, initial_state, mcp_game, τs_observed)
        solution = solve_mcp_game(mcp_game, initial_state, context_state_estimation)
        if solution.status != :solved
            return 1e10, solution
        end
        observed_τs_solution = observe_trajectory(solution.xs)
        norm_sqr(vcat(τs_observed...) - observed_τs_solution), solution
    end

    infeasible_counter = 0
    solving_info = []
    context_state_estimation = initial_estimation
    i_ = 0
    time_exec = 0
    solving_status = []
    
    for i in 1:max_grad_steps
        i_ = i
        
        loss_for_grad = c -> loss(c, initial_state, mcp_game, τs_observed)
        grad_step_time = @elapsed (l_val, solution_val), pb = Zygote.pullback(loss_for_grad, context_state_estimation)
        push!(solving_status, solution_val.status)

        if isnothing(pb)
            @info "Gradient calculation failed (inner solver may have failed). Stopping optimization at iteration: "*string(i)
            break
        end
        gradient = pb((1.0, nothing))[1]
        if isnothing(gradient)
             @info "Gradient is `nothing`. This can happen if the inner solver fails. Stopping optimization at iteration: "*string(i)
             @info "Inner solver status: " * string(solution_val.status)
            break
        end

        time_exec += grad_step_time
        objective_grad = gradient
        clamp!(objective_grad, -50, 50)
        objective_update = lr * objective_grad
        println("objective_update norm: ", norm(objective_update))
        println("termination condition: ", norm(objective_update)/norm(context_state_estimation))
        
        if norm(objective_grad)/norm(context_state_estimation) < 1e-3
            @info "Inner iteration terminates at iteration: "*string(i)
            break
        elseif solution_val.status != PATHSolver.MCP_Solved
            infeasible_counter += 1
            if infeasible_counter >= 4
                @info "Inner iteration reached the maximal infeasible steps"
                break
            end
        else
            infeasible_counter = 0
        end
        context_state_estimation -= objective_update
        println("context_state_estimation: ", context_state_estimation)
    end
    (; context_state_estimation, last_solution, i_, solving_info, time_exec, solving_status)
end

