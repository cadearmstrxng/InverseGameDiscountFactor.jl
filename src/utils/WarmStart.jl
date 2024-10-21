using TrajectoryGamesBase:
    GeneralSumCostStructure, TrajectoryGameCost
using Infiltrator

export warm_start


struct NullEnv end

function TrajectoryGamesBase.get_constraints(env::NullEnv, ii)
    function null_constraints(x)
        []
    end
    null_constraints
end

function warm_start_game(num_players;
    environment,
    observation_model = (; σ = 0.0, expected_observation = identity),
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),)
    )

    cost = let

        function warm_start_cost_for_player(i, x, y, T, state_size)
            c = []
            for t in 1:T
                # @infiltrate
                push!(c,norm_sqr(observation_model.expected_observation(x[t+1][Block(i)]) - y[Block(t)][((i-1)*state_size+1):(i*state_size)]))
            end

            return sum(c)
        end

        function warm_start_cost(x,y)
            T = convert(Int64,size(y)[1]/size(y[Block(1)])[1])
            state_size = size(x[1][Block(1)])[1]
            num_players = convert(Int64,size(y[Block(1)])[1]//state_size)

            [warm_start_cost_for_player(i, x, y, T, state_size) for i in 1:num_players]
        end
        TrajectoryGameCost(warm_start_cost, GeneralSumCostStructure())

    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    game = TrajectoryGame(
        dynamics,
        cost,
        environment,
        nothing,
    )
    CollisionAvoidanceGame(game)
end

"""
This file contains problem formulation part of the MCP game solver code: casting open-loop
Nash games as mixed complementarity problems (MCPs)
A more optimized implementation of this solver is available at: 
https://github.com/JuliaGameTheoreticPlanning/MCPTrajectoryGameSolver.jl
"""

struct WarmStartMCPGame{T1<:TrajectoryGame,T2<:ParametricMCPs.ParametricMCP,T3,T4}
    game::T1
    parametric_mcp::T2
    index_sets::T3
    horizon::T4
end

function WarmStartMCPGame(game, observed_trajectory, horizon)
    num_player = num_players(game)
    state_dimension = state_dim(game.dynamics)
    control_dimension = control_dim(game.dynamics)
    problem_size = horizon * (state_dimension + control_dimension)
    state_block_dimensions = [state_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]
    control_block_dimensions = [control_dim(game.dynamics.subsystems[ii]) for ii in 1:num_player]

    x0, z = let
        @variables(
            x0[1:state_dimension],
            z[1:problem_size]
        ) .|> scalarize
    end

    # convert to block structure
    x0 = BlockVector(x0, state_block_dimensions)

    # @infiltrate

    function trajectory_from_flattened_decision_variables(flattened_z)
        states = let
            future_states = eachcol(
                reshape(flattened_z[1:(state_dimension * horizon)], state_dimension, horizon),
            )
            map(Iterators.flatten(([x0], future_states))) do x_joint
                BlockVector(collect(x_joint), state_block_dimensions)
            end
        end

        control_inputs = let
            controls = eachcol(
                reshape(
                    flattened_z[(state_dimension * horizon + 1):end],
                    control_dimension,
                    horizon,
                ),
            )
            map(controls) do u_joint
                BlockVector(collect(u_joint), control_block_dimensions)
            end
        end
        (states, control_inputs)
    end

    (xs, us) = trajectory_from_flattened_decision_variables(z)
    (x_idx, u_idx) = trajectory_from_flattened_decision_variables(1:problem_size)
    # @infiltrate
    x_idx = x_idx[2:end]
    # @infiltrate
    τ_idx_set = let
        map(1:num_player) do ii
            x_idx_single_player = [idx[Block(ii)] for idx in x_idx]
            u_idx_single_player = [idx[Block(ii)] for idx in u_idx]
            [reduce(vcat, x_idx_single_player); reduce(vcat, u_idx_single_player)]
        end
    end

    cost_per_player = game.cost(xs, observed_trajectory) .|> scalarize
    # @infiltrate

    lb = Float64[]
    ub = Float64[]
    private_constraints = []
    num_private_constraints = []
    f = let
        @variables(f[1:problem_size]) |> only |> scalarize
    end

    private_constraint_per_player = let
        map(1:num_player) do ii
            environment_constraints = get_constraints(game.env, ii)
            subdynamics = game.dynamics.subsystems[ii]
            private_inequality_constraints = let
                state_box_constraints = get_constraints_from_box_bounds(state_bounds(subdynamics))
                control_box_constraints =
                    get_constraints_from_box_bounds(control_bounds(subdynamics))

                ec = mapreduce(x -> environment_constraints(x[Block(ii)]), vcat, xs[2:end])
                sc = mapreduce(x -> state_box_constraints(x[Block(ii)]), vcat, xs[2:end])
                cc = mapreduce(x -> control_box_constraints(x[Block(ii)]), vcat, us[1:end])
                [ec; sc; cc]
                # @infiltrate
            end
            append!(lb, fill(0.0, length(private_inequality_constraints)))
            append!(ub, fill(Inf, length(private_inequality_constraints)))
            append!(private_constraints, private_inequality_constraints)
            append!(num_private_constraints, length(private_inequality_constraints))
        end
    end

    # set up the gradient of lagrangian
    private_duals = let
        @variables(μ[1:sum(num_private_constraints)]) |> only |> scalarize
    end
    for ii in 1:num_player
        dual_offset = (ii > 1) ? sum(num_private_constraints[1:(ii - 1)]) : 0
        # @infiltrate
        local_lagrangian =
            cost_per_player[ii] -
            private_duals[(dual_offset + 1):(dual_offset + num_private_constraints[ii])]' *
            private_constraints[(dual_offset + 1):(dual_offset + num_private_constraints[ii])]
        f[τ_idx_set[ii]] = Symbolics.gradient(local_lagrangian, z[τ_idx_set[ii]])
    end
    private_inequality_idx_set = collect((problem_size + 1):(problem_size + length(private_duals)))

    lb = vcat(fill(-Inf, problem_size), lb)
    ub = vcat(fill(Inf, problem_size), ub)
    # append private duals to z
    z = vcat(z, private_duals)
    f = vcat(f, private_constraints)

    #shared dynamics equality constraints
    shared_dynamics_equality_constraints = mapreduce(vcat, 1:horizon) do t
        game.dynamics(xs[t], us[t], t) .- xs[t + 1]
    end
    shared_dynamics_duals = let
        @variables(γ[1:length(shared_dynamics_equality_constraints)]) |> only |> scalarize
    end
    shared_dynamics_jacobian = Symbolics.sparsejacobian(shared_dynamics_equality_constraints, z)
    f -= shared_dynamics_jacobian' * shared_dynamics_duals
    z = vcat(z, shared_dynamics_duals)
    f = vcat(f, shared_dynamics_equality_constraints)
    lb = vcat(lb, fill(-Inf, length(shared_dynamics_equality_constraints)))
    ub = vcat(ub, fill(Inf, length(shared_dynamics_equality_constraints)))
    shared_equality_idx_set = collect(
        (private_inequality_idx_set[end] + 1):(private_inequality_idx_set[end] + length(
            shared_dynamics_duals,
        )),
    )

    # coupled inequality constraints
    if !isnothing(game.coupling_constraints)
        coupled_inequality_constraints = game.coupling_constraints(xs[2:end], us)
        coupled_duals = let
            @variables(λ[1:length(coupled_inequality_constraints)]) |> only |> scalarize
        end
        coupled_constraints_jacobian = Symbolics.sparsejacobian(coupled_inequality_constraints, z)
        f -= coupled_constraints_jacobian' * coupled_duals

        z = vcat(z, coupled_duals)
        f = vcat(f, coupled_inequality_constraints)
        lb = vcat(lb, fill(0.0, length(coupled_inequality_constraints)))
        ub = vcat(ub, fill(Inf, length(coupled_inequality_constraints)))
        shared_inequality_idx_set = collect(
            (shared_equality_idx_set[end] + 1):(shared_equality_idx_set[end] + length(
                coupled_duals,
            )),
        )
    else
        shared_inequality_idx_set = nothing
    end

    θ = [x0;nothing]
    index_sets = (;
        τ_idx_set,
        private_inequality_idx_set,
        shared_equality_idx_set,
        shared_inequality_idx_set,
    )

    # F vector parametriztion
    fill_F! = let
        F! = Symbolics.build_function(f, [z; θ]; expression = Val{false})[2]
        (vals, z, θ) -> F!(vals, [z; θ])
    end

    # J matrix: jacobian of F vector
    J = Symbolics.sparsejacobian(f, z)

    fill_J! = let
        J! = Symbolics.build_function(J, [z; θ]; expression = Val{false})[2]
        rows, cols, _ = findnz(J)
        constant_entries = ParametricMCPs.get_constant_entries(J, z)
        ParametricMCPs.SparseFunction(rows, cols, size(J), constant_entries) do vals, z, θ
            J!(vals, [z; θ])
        end
    end

    # J_params: jacobian of F vector w.r.t. the parameters (i.e. the context_state)
    J_params = Symbolics.sparsejacobian(f, θ)

    fill_J_params! = let
        J_params! = Symbolics.build_function(J_params, [z; θ]; expression = Val{false})[2]
        rows, cols, _ = findnz(J_params)
        ParametricMCPs.SparseFunction(rows, cols, size(J_params)) do vals, z, θ
            J_params!(vals, [z; θ])
        end
    end

    parameter_dimension = length(θ)
    parametric_mcp =
        ParametricMCPs.ParametricMCP(fill_F!, fill_J!, fill_J_params!, lb, ub, parameter_dimension)

    MCPGame(game, parametric_mcp, index_sets, horizon)
end

#== MCP TrajectoryGame Solver ==#

function MCPCoupledOptimizationSolverWarmStart(game::TrajectoryGame, observed_trajectory, horizon)
    mcp_game = WarmStartMCPGame(game, observed_trajectory, horizon)
    MCPCoupledOptimizationSolver(mcp_game)
end


function warm_start(y, initial_state, horizon; observation_model = (; σ = 0.0, expected_observation = identity), 
                    num_players = 2)

    environment = NullEnv()
    game = warm_start_game(num_players; environment, observation_model = observation_model)

    # @infiltrate

    turn_length = 2
    solver = MCPCoupledOptimizationSolverWarmStart(game.game, y, horizon)
    mcp_game = solver.mcp_game

    @infiltrate

    solve_mcp_game(mcp_game, initial_state, nothing; verbose = true)

end


