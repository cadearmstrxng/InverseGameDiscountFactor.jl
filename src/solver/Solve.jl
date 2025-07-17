"""
This file contains solving part of the MCP game solver code.
A more optimized implementation of this solver is available at: 
https://github.com/JuliaGameTheoreticPlanning/MCPTrajectoryGameSolver.jl
"""

function solve_mcp_game_rh(
    mcp_game::MCPGame,
    x0,
    context_state;
    initial_guess = nothing,
    verbose = false,
    lr = 1e-3,
    total_horizon = 10
)
    (; game, parametric_mcp, index_sets, horizon) = mcp_game
    (; dynamics) = game
    final_primals_xs = [[] for _ in 1:num_players(game)]
    final_primals_us = [[] for _ in 1:num_players(game)]

    control_block_dimensions, state_dimensions, dummy_strategy, controls_offset, variables, status, info = ChainRulesCore.ignore_derivatives() do
        cbd = [control_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
        sd = [state_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
        dummy_strategy = (x, t) -> BlockVector(zeros(sum(cbd)), cbd)
        offset = [sd[ii]*horizon+1 for ii in 1:num_players(game)]
        cbd, sd, dummy_strategy, offset, nothing, nothing, nothing
    end

    num_successful_solves = 0

    z = ChainRulesCore.ignore_derivatives() do
        #initial guess
        if !isnothing(initial_guess)
            z = initial_guess.variables
            verbose && @info "Warm-started with the previous solution."
        else
            x0_value = ForwardDiff.value.(x0)
            z = zeros(length(parametric_mcp.lower_bounds))
            xs = rollout(dynamics, dummy_strategy, x0_value, horizon + 1).xs[2:end]
            xs = reduce(vcat, xs)
            z[1:(sum(state_dimensions) * horizon)] = xs
        end
        z
    end
    for i in 1:total_horizon
        θ = [x0; context_state]
        variables, status, info = ParametricMCPs.solve(
            parametric_mcp,
            θ;
            initial_guess = z,
            verbose,
            cumulative_iteration_limit = 100_000,
            convergence_tolerance = 1e-3
        ) 
        if status != PATHSolver.MCP_Solved
            variables_, status_, info_ = ParametricMCPs.solve(
            parametric_mcp,
            θ;
            verbose,
            convergence_tolerance = 1e-3
            )
            if status_ == PATHSolver.MCP_Solved
                variables = variables_
                status = status_
                info = info_
            end
        end
        if status == PATHSolver.MCP_Solved
            num_successful_solves += 1
        end
        ChainRulesCore.ignore_derivatives() do
            @info "time step: $i, $status"
        end
        primals = map(1:num_players(game)) do ii
            variables[index_sets.τ_idx_set[ii]]
        end
        final_primals_xs = map(1:num_players(game)) do ii
            push!(final_primals_xs[ii], primals[ii][1:state_dimensions[ii]])
        end
        if i != total_horizon
            final_primals_us = map(1:num_players(game)) do ii
                push!(final_primals_us[ii], primals[ii][controls_offset[ii]:controls_offset[ii] + control_block_dimensions[ii] - 1])
            end
        end
        z = ChainRulesCore.ignore_derivatives() do
            inital_guess_strategy = (x, t) -> (t < horizon) ?
                BlockVector(
                    vcat([ForwardDiff.value.(primals[ii][controls_offset[ii] + (t-1)*control_block_dimensions[ii]:controls_offset[ii]-1+ t*control_block_dimensions[ii]]) for ii in 1:num_players(game)]...),
                    control_block_dimensions) : dummy_strategy(x, t)
            xs = rollout(dynamics, inital_guess_strategy, ForwardDiff.value.(x0), horizon + 1).xs[2:end]
            xs = reduce(vcat, xs)
            z[1:(sum(state_dimensions) * horizon)] = xs
            z = copy(ForwardDiff.value.(variables))
        end
        placeholder_strategy = (x, t) -> (t < horizon) ?
            BlockVector(vcat([primals[ii][controls_offset[ii] + (t-1)*control_block_dimensions[ii]:controls_offset[ii]-1+ t*control_block_dimensions[ii]] for ii in 1:num_players(game)]...), control_block_dimensions) : dummy_strategy(x, t)
        x0 = rollout(dynamics, placeholder_strategy, x0, 3).xs[2]
    end
    primals = map(1:num_players(game)) do ii
        vcat(vcat(final_primals_xs[ii]...), vcat(final_primals_us[ii]...))
    end
    success_ratio = total_horizon > 0 ? num_successful_solves / total_horizon : 1.0
    (; primals, variables, status, info, success_ratio)
end

function solve_mcp_game(
    mcp_game::MCPGame,
    x0,
    context_state;
    initial_guess = nothing,
    verbose = false,
    lr = 1e-3,
    total_horizon = 10,
    maxiter = 100_000
)
    (; game, parametric_mcp, index_sets, horizon) = mcp_game
    (; dynamics) = game
    final_primals_xs = [[] for _ in 1:num_players(game)]
    final_primals_us = [[] for _ in 1:num_players(game)]

    control_block_dimensions, state_dimensions, dummy_strategy, controls_offset, variables, status, info = ChainRulesCore.ignore_derivatives() do
        cbd = [control_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
        sd = [state_dim(dynamics.subsystems[ii]) for ii in 1:num_players(game)]
        dummy_strategy = (x, t) -> BlockVector(zeros(sum(cbd)), cbd)
        offset = [sd[ii]*horizon+1 for ii in 1:num_players(game)]
        cbd, sd, dummy_strategy, offset, nothing, nothing, nothing
    end

    z = ChainRulesCore.ignore_derivatives() do
        #initial guess
        if !isnothing(initial_guess)
            z = initial_guess.variables
            verbose && @info "Warm-started with the previous solution."
        else
            x0_value = ForwardDiff.value.(x0)
            z = zeros(length(parametric_mcp.lower_bounds))
            xs = rollout(dynamics, dummy_strategy, x0_value, horizon + 1).xs[2:end]
            xs = reduce(vcat, xs)
            z[1:(sum(state_dimensions) * horizon)] = xs
        end
        z
    end

    θ = [x0; context_state]
    variables, status, info = ParametricMCPs.solve(
        parametric_mcp,
        θ;
        initial_guess = z,
        verbose,
        cumulative_iteration_limit = maxiter,
        convergence_tolerance = 1e-3
    ) 
    if status != PATHSolver.MCP_Solved
        variables_, status_, info_ = ParametricMCPs.solve(
        parametric_mcp,
        θ;
        verbose,
        convergence_tolerance = 1e-3,
        cumulative_iteration_limit = maxiter
        )
        if status_ == PATHSolver.MCP_Solved
            variables = variables_
            status = status_
            info = info_
        end
    end
    primals = map(1:num_players(game)) do ii
        variables[index_sets.τ_idx_set[ii]]
    end
    (; primals, variables, status, info, success_ratio = status == PATHSolver.MCP_Solved ? 1.0 : 0.0)
end