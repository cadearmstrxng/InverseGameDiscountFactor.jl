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
    lr = 1e-3
)
    solution = MixedComplementarityProblems.solve(
        MixedComplementarityProblems.InteriorPoint(),
        mcp_game.mcp,
        context_state;
        verbose = verbose,
    )

    interpret_variables(solution, mcp_game)   
end

function interpret_variables(sol, game::MCPGame; debug::Bool = false)
    dims = get_dimensions(game)

    # Extract states and controls for each player
    xs = map(1:game.horizon) do t
        mapreduce(vcat, 1:dims.n_players) do ii
            sol.x[dims.player_xs[ii][t]]
        end
    end
    
    us = map(1:game.horizon) do t
        mapreduce(vcat, 1:dims.n_players) do ii
            sol.x[dims.x_size .+ dims.player_us[ii][t]]
        end
    end

    # Extract duals
    player_λs = map(1:dims.n_players) do ii
        sum(game.n_inequality_constraints[1:ii-1])+1:sum(game.n_inequality_constraints[1:ii])
    end
    player_μs = map(1:dims.n_players) do ii
        sum(game.n_equality_constraints[1:ii-1])+1:sum(game.n_equality_constraints[1:ii])
    end

    # Extract equality multipliers (μ) for each player
    μs = BlockVector(
        mapreduce(vcat, 1:dims.n_players) do ii
            sol.x[dims.x_size + dims.u_size .+ player_μs[ii]]
        end,
        game.n_equality_constraints
    )

    # Extract inequality multipliers (λ) for each player
    λs = BlockVector(
        mapreduce(vcat, 1:dims.n_players) do ii
            sol.y[player_λs[ii]]
        end,
        game.n_inequality_constraints
    )

    # Extract shared inequality multipliers
    λ_sh = sol.y[sum(game.n_inequality_constraints)+1:end]

    slack = sol.s

    return (; xs, us, μs, λs, λ_sh, slack, status = sol.status)
end