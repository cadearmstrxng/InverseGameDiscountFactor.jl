module ExperimentUtils

using Profile
using Timers
using ProfileView: @profview
using BlockArrays: blocksizes
# using InverseGameDiscountFactor

include("./GameUtils.jl")
include("../src/InverseGameDiscountFactor.jl")

function run_profile_test(num_iters, mcp_game, init, observed_forward_solution;
    full_state = true, graph = true, verbose = true
)   
    

    
    for _ in num_iters
        Timers.tic()
        method_sol = InverseGameDiscountFactor.solve_myopic_inverse_game(
            mcp_game,
            observed_forward_solution,
            init.observation_model,
            (3, 3);
            hidden_state_guess = init.game_parameters,
            max_grad_steps = 200,
            retries_on_divergence = 3,
            verbose = false,
            total_horizon = init.horizon
        )
        Timers.toc(true)
    end
end


function generate_flame_graph(;full_state = true)

    init = GameUtils.init_crosswalk_game(
        full_state;
        myopic = true,
        horizon = 10
    )

    mcp_game = InverseGameDiscountFactor.MCPCoupledOptimizationSolver(
        init.game_structure.game,
        init.horizon,
        blocksizes(init.game_parameters, 1)
    ).mcp_game

    forward_solution = InverseGameDiscountFactor.reconstruct_solution(
            InverseGameDiscountFactor.solve_mcp_game(
                mcp_game,
                init.initial_state,
                init.game_parameters;
                verbose = false,
                maxiter = 100_000
                ),
            init.game_structure.game,
            init.horizon
        )

    observed_forward_solution = GameUtils.observe_trajectory(forward_solution, init)

    run_profile_test(1, mcp_game, init, observed_forward_solution)
    Profile.clear()
    @profview run_profile_test(1, mcp_game, init, observed_forward_solution) # TODO: do zygote reverse mode AD?
    # Redirect Profile.print output to a file
    open("flamegraph_output.txt", "w") do io
        redirect_stdout(io) do
            Profile.print(format=:flat, C=true, combine=true, sortedby=:time, maxdepth=20, mincount=0, noisefloor=0.0)
        end
    end
end

end