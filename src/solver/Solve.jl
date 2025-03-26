"""
This file contains solving part of the MCP game solver code.
A more optimized implementation of this solver is available at: 
https://github.com/JuliaGameTheoreticPlanning/MCPTrajectoryGameSolver.jl
"""

function solve_mcp_game(
    mcp_game::MCPGame,
    x0,
    context_state;
    initial_guess = nothing,
    verbose = false,
    lr = 1e-3,
    rh_horizon = 5
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
    @infiltrate
    for i in 1:horizon - rh_horizon
        θ = [x0; context_state]
        variables, status, info = ParametricMCPs.solve(
            parametric_mcp,
            θ;
            initial_guess = z,
            verbose,
            cumulative_iteration_limit = 100_000,
            proximal_perturbation = 1e-2,
            use_basics = true,
            use_start = true,
            lr = lr
        ) # change this to David's package -> MixedComplementarityProblem.PrimalDualMCP (mcp.jl)

        primals = map(1:num_players(game)) do ii
            variables[index_sets.τ_idx_set[ii]]
        end
        final_primals_xs = map(1:num_players(game)) do ii
            push!(final_primals_xs[ii], primals[ii][1:state_dimensions[ii]])
        end
        final_primals_us = map(1:num_players(game)) do ii
            push!(final_primals_us[ii], primals[ii][controls_offset[ii]:controls_offset[ii] + control_block_dimensions[ii] - 1])
        end

        z = ChainRulesCore.ignore_derivatives() do
            first_action = (x, t) -> BlockVector(vcat([primals[ii][state_dimensions[ii]*horizon + 1:state_dimensions[ii]*horizon + sum(control_block_dimensions[ii])] for ii in 1:num_players(game)]...),
                control_block_dimensions)
            xs = rollout(dynamics, first_action, x0, horizon + 1).xs[2:end] # may need to increase to horizon + 2 if size doesn't fit
            x0 = xs[1]
            xs = reduce(vcat, xs)
            z[1:(sum(state_dimensions) * horizon)] = ForwardDiff.value.(xs)
            z
        end
    end

    primals = map(1:num_players(game)) do ii
        vcat(vcat(final_primals_xs[ii]...), vcat(final_primals_us[ii]...))
    end
    (; primals, variables, status, info)
end

function TrajectoryGamesBase.solve_trajectory_game!(
    solver::MCPCoupledOptimizationSolver,
    game::TrajectoryGame{<:ProductDynamics},
    initial_state,
    strategy;
    verbose = false,
    solving_info = nothing
)
    problem = solver.mcp_game
    if !isnothing(strategy.last_solution) && strategy.last_solution.status == PATHSolver.MCP_Solved
        solution = solve_mcp_game(
            solver.mcp_game,
            initial_state,
            strategy.context_state;
            initial_guess = strategy.last_solution,
            verbose
        )
    else
        solution = solve_mcp_game(solver.mcp_game, initial_state, strategy.context_state; verbose)
    end
    if !isnothing(solving_info)
        push!(solving_info, solution.info)
    end
    # warm-start only when the last solution is valid
    if solution.status == PATHSolver.MCP_Solved
        strategy.last_solution = solution
    end
    strategy.solution_status = solution.status

    rng = Random.MersenneTwister(1)

    horizon = solver.mcp_game.horizon
    num_player = num_players(game)
    state_block_dimensions = [state_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]
    control_block_dimensions = [control_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]

    substrategies = let
        map(1:num_player) do ii
            xs = [
                [initial_state[Block(ii)]]
                collect.(
                    eachcol(
                        reshape(
                            solution.primals[ii][1:(horizon * state_block_dimensions[ii])],
                            state_block_dimensions[ii],
                            :,
                        ),
                    )
                )
            ]
            us =
                collect.(
                    eachcol(
                        reshape(
                            solution.primals[ii][(horizon * state_block_dimensions[ii] + 1):end],
                            control_block_dimensions[ii],
                            :,
                        ),
                    )
                )

            LiftedTrajectoryStrategy(ii, [(; xs, us)], [1], nothing, rng, Ref(0))
        end
    end
    JointStrategy(substrategies)
end
