#=== Mixed-Complementarity Problem Formulation (MCP) ===#
"""
This file contains problem formulation part of the MCP game solver code: casting open-loop
Nash games as mixed complementarity problems (MCPs)
A more optimized implementation of this solver is available at: 
https://github.com/JuliaGameTheoreticPlanning/MCPTrajectoryGameSolver.jl
"""

struct MCPGame{T1, T2}
    game::T1
    mcp::T2
    horizon::Int
    n_equality_constraints::Vector{Int}
    n_inequality_constraints::Vector{Int}
    n_shared_inequality_constraints::Int
    player_lagrangians_L::Vector{Symbolics.Num}
    initial_state::Vector{Float64}
end
debug = true 

function MCPGame(game::TrajectoryGame, horizon::Int, initial_conditions::Vector{Float64}, context_state_block_dimensions::Vector{Int})
    dynamics = game.dynamics
    costs = game.cost
    dims = get_dimensions(game, horizon)

    n_eq_constr = zeros(Int, dims.n_players)
    n_ineq_constr = zeros(Int, dims.n_players)
    n_shared_ineq_constr = 0
    
    for ii in 1:dims.n_players
        # Dynamics constraints
        n_eq_constr[ii] += dims.state_dims[ii] * (horizon - 1)
        # Initial conditions
        n_eq_constr[ii] += dims.state_dims[ii]

        # Inequality constraints
        # Environment, state, control box constraints counted later (sometimes unbounded)
    end
    # Grab shared inequality constraints from game's coupling constraints - Assume number is not a function of time
    n_shared_ineq_constr = isnothing(game.coupling_constraints) ? 0 : length(game.coupling_constraints(BlockVector(zeros(x_size), [x_size]), BlockVector(zeros(u_size), [u_size]), 1)) * horizon

    # Equality constraints
    equality_constr_funcs = map(1:dims.n_players) do ii
        function(_z, _λ; θ = nothing)
            x = _z[1:dims.x_size]
            u = _z[dims.x_size+1:dims.x_size+dims.u_size]
            
            dyn_constr = mapreduce(vcat, 1:horizon-1) do t
                game.dynamics.subsystems[ii](x[dims.player_xs[ii][t]], u[dims.player_us[ii][t]], t) - x[dims.player_xs[ii][t+1]]
            end
            init_constr = x[dims.player_xs[ii][1]] - initial_conditions[dims.player_xs[ii][1]]
            return vcat(dyn_constr, init_constr)
        end
    end

    # Inequality constraints
    inequality_constr_funcs = map(1:dims.n_players) do ii
        environment_constraints_gen = get_constraints(game.env, ii)
        subdynamics = dynamics.subsystems[ii]
        state_box_constraints_gen = get_constraints_from_box_bounds(state_bounds(subdynamics))
        control_box_constraints_gen = get_constraints_from_box_bounds(control_bounds(subdynamics))

        n_ineq_constr[ii] += length(environment_constraints_gen(BlockVector(zeros(dims.state_dims[ii]), [dims.state_dims[ii]]))) * horizon
        n_ineq_constr[ii] += length(state_box_constraints_gen(BlockVector(zeros(dims.state_dims[ii]), [dims.state_dims[ii]]))) * horizon
        n_ineq_constr[ii] += length(control_box_constraints_gen(BlockVector(zeros(dims.control_dims[ii]), [dims.control_dims[ii]]))) * horizon

        function(_z, _λ; θ = nothing)
            x = _z[1:dims.x_size]
            u = _z[dims.x_size+1:dims.x_size+dims.u_size]
            
            ec = mapreduce(vcat, 1:horizon) do t
                environment_constraints_gen(BlockVector(x[dims.player_xs[ii][t]], [dims.state_dims[ii]]))
            end
            sc = mapreduce(vcat, 1:horizon) do t
                state_box_constraints_gen(BlockVector(x[dims.player_xs[ii][t]], [dims.state_dims[ii]]))
            end
            cc = mapreduce(vcat, 1:horizon) do t
                control_box_constraints_gen(BlockVector(u[dims.player_us[ii][t]], [dims.control_dims[ii]]))
            end
            return vcat(ec, sc, cc)
        end
    end
    !debug || println("[ProblemFormulation] Inequality constraints solved")

    # Shared inequality constraints
    local shared_inequality_constr_eval_func::Function
    if game.coupling_constraints !== nothing && n_shared_ineq_constr > 0
        let captured_game_cc = game.coupling_constraints 
            shared_inequality_constr_eval_func = (_z_arg, _λ_arg; θ = nothing) -> begin
                mapreduce(vcat, 1:horizon) do t
                    primal_vars_z = _z_arg[1:(x_size+u_size)]
                    x_slice = _z_arg[1:x_size]
                    u_slice = _z_arg[x_size+1 : x_size+u_size]
                    temp_constraints = captured_game_cc(_z_arg[1:(x_size+u_size)], _λ_arg, t)
                    return (temp_constraints isa Vector{Symbolics.Num} ? temp_constraints : convert(Vector{Symbolics.Num}, temp_constraints))
                end
            end
        end
    else
        shared_inequality_constr_eval_func = (_z_arg, _λ_arg; θ = nothing) -> Vector{Symbolics.Num}()
    end
    !debug || println("[ProblemFormulation] Shared inequality constraints solved")

    all_player_lagrangians_L = Vector{Symbolics.Num}()
    lagrangian_grads = let
        # Define symbolic variables for the Lagrangian scope
        @variables z_L[1:dims.x_size+dims.u_size+sum(n_eq_constr)] λ_L[1:sum(n_ineq_constr)] λ_sh_L[1:n_shared_ineq_constr] θ_L[1:sum(context_state_block_dimensions)]
        z_L = Symbolics.scalarize(z_L)
        λ_L = Symbolics.scalarize(λ_L)
        λ_sh_L = Symbolics.scalarize(λ_sh_L)
        θ_L = Symbolics.scalarize(θ_L)

        xs_L = map(1:horizon) do t
            BlockVector(z_L[(t-1)*sum(dims.state_dims)+1:t*sum(dims.state_dims)], dims.state_dims)
        end
        us_L = map(1:horizon) do t
            BlockVector(z_L[dims.x_size + (t-1)*sum(dims.control_dims)+1:dims.x_size + t*sum(dims.control_dims)], dims.control_dims)
        end
        μs_L = z_L[dims.x_size+dims.u_size+1 : dims.x_size+dims.u_size+sum(n_eq_constr)] 

        player_λ_indices = mapreduce(vcat, 1:dims.n_players) do ii
            [sum(n_ineq_constr[1:ii-1])+1:sum(n_ineq_constr[1:ii])]
        end
        player_μ_indices = mapreduce(vcat, 1:dims.n_players) do ii
            [sum(n_eq_constr[1:ii-1])+1:sum(n_eq_constr[1:ii])]
        end

        context_state_L = BlockVector(θ_L, context_state_block_dimensions)
        player_costs = game.cost(xs_L, us_L, context_state_L)

        gradient_functions = map(1:dims.n_players) do ii
            L_ii = let
                cost_term = player_costs[ii]
                eq_constr_exprs = equality_constr_funcs[ii](z_L, λ_L; θ = context_state_L)
                eq_term = dot(μs_L[player_μ_indices[ii]], eq_constr_exprs)

                ineq_constr_exprs = inequality_constr_funcs[ii](z_L, λ_L; θ = context_state_L) 
                ineq_term_player = if !isempty(ineq_constr_exprs)
                    dot(λ_L[player_λ_indices[ii]], ineq_constr_exprs)
                else
                    0
                end
                
                ineq_term_shared = 0
                if n_shared_ineq_constr > 0
                    shared_ineq_exprs = shared_inequality_constr_eval_func(z_L, λ_L; θ = context_state_L)
                    ineq_term_shared = dot(λ_sh_L, shared_ineq_exprs)
                end
                cost_term - eq_term - ineq_term_player - ineq_term_shared 
            end
            
            !debug || println("[ProblemFormulation] Player $ii Lagrangian solved")
            push!(all_player_lagrangians_L, L_ii)

            player_x_vars = reduce(vcat, map(t -> z_L[dims.player_xs[ii][t]], 1:horizon))
            player_u_vars = reduce(vcat, map(t -> z_L[dims.x_size .+ dims.player_us[ii][t]], 1:horizon))
            
            ∇x_L_ii = Symbolics.gradient(L_ii, player_x_vars)
            ∇u_L_ii = Symbolics.gradient(L_ii, player_u_vars)
            # ∇x_L_ii = Symbolics.sparsejacobian([L_ii], player_x_vars)[1]
            # ∇u_L_ii = Symbolics.sparsejacobian([L_ii], player_u_vars)[1]
            !debug || println("[ProblemFormulation] Player $ii gradient solved")
            
            function(_z_arg, _λ_arg; θ = nothing) 
                all_L_sym_vars = vcat(z_L, λ_L, λ_sh_L, θ_L) 
                
                _λ_L_part = _λ_arg[1:sum(n_ineq_constr)]
                _λ_sh_L_part = _λ_arg[sum(n_ineq_constr)+1 : sum(n_ineq_constr)+n_shared_ineq_constr]
                all_arg_runtime_vals = vcat(_z_arg, _λ_L_part, _λ_sh_L_part, θ) 

                if length(all_L_sym_vars) != length(all_arg_runtime_vals)
                    println("--- ERROR in lagrangian_grads substitution ---")
                    println("Symbolic vars (all_L_sym_vars), count: ", length(all_L_sym_vars))
                    println("Runtime vals (all_arg_runtime_vals), count: ", length(all_arg_runtime_vals))
                    println("Breakdown of symbolic vars:")
                    println("  z_L: ", length(z_L))
                    println("  λ_L: ", length(λ_L))
                    println("  λ_sh_L: ", length(λ_sh_L))
                    println("Breakdown of runtime args (_z_arg, _λ_arg parts):")
                    println("  _z_arg: ", length(_z_arg))
                    println("  _λ_L_part (from _λ_arg): ", length(_λ_L_part))
                    println("  _λ_sh_L_part (from _λ_arg): ", length(_λ_sh_L_part))
                    println("--------------------------------------------")
                    error("LAGRANGIAN_GRADS_SUBSTITUTION_ERROR: Mismatch between symbolic variable count (" * string(length(all_L_sym_vars)) * ") and value count (" * string(length(all_arg_runtime_vals)) * ") for substitution. Check z_L, λ_L, λ_sh_L against _z_arg and _λ_arg partitioning.")
                end

                subs_map = Dict(zip(all_L_sym_vars, all_arg_runtime_vals))

                gx = Symbolics.substitute.(∇x_L_ii, Ref(subs_map))
                gu = Symbolics.substitute.(∇u_L_ii, Ref(subs_map))
                
                vcat(gx, gu)
            end
        end
        gradient_functions 
    end

    G = function(_z_mcp, _λ_mcp; θ = nothing) 
        lag_grad_components = mapreduce(vcat, lagrangian_grads) do grad_func_for_player
            grad_func_for_player(_z_mcp, _λ_mcp; θ = θ) 
        end
        eq_constr_components = mapreduce(vcat, equality_constr_funcs) do eq_func
            eq_func(_z_mcp, _λ_mcp) 
        end
        vcat(lag_grad_components, eq_constr_components)
    end
    !debug || println("[ProblemFormulation] G solved")
    
    H = function(_z_mcp, _λ_mcp; θ = nothing) 
        ineq_player_components = mapreduce(vcat, inequality_constr_funcs) do ineq_func
            ineq_func(_z_mcp, _λ_mcp; θ = θ)
        end

        shared_ineq_components = Vector{Symbolics.Num}()
        if n_shared_ineq_constr > 0
            shared_ineq_components = shared_inequality_constr_eval_func(_z_mcp, _λ_mcp; θ = θ)
        end
        
        vcat(ineq_player_components, shared_ineq_components)
    end
    !debug || println("[ProblemFormulation] H solved")
    # @infiltrate
    
    !debug || println("[ProblemFormulation] Starting MCP initialization...")
    start_time = time()
    mcp = MixedComplementarityProblems.PrimalDualMCP(
        G,
        H;
        unconstrained_dimension = dims.x_size + dims.u_size + sum(n_eq_constr), 
        constrained_dimension = sum(n_ineq_constr) + n_shared_ineq_constr,
        parameter_dimension = sum(context_state_block_dimensions),
        compute_sensitivities = true,
    )
    solve_time = time() - start_time
    !debug || println("[ProblemFormulation] MCP initialized in $(round(solve_time, digits=2)) seconds")
    
    return MCPGame(game, mcp, horizon, n_eq_constr, n_ineq_constr, n_shared_ineq_constr, all_player_lagrangians_L, initial_conditions)
end

#== MCP TrajectoryGame Solver ==#

struct MCPCoupledOptimizationSolver
    mcp_game::MCPGame
end

function MCPCoupledOptimizationSolver(game::TrajectoryGame, horizon, initial_conditions, context_state_block_dimensions; verbose = false)
    verbose || print("initializing mcp game ... ")
    mcp_game = MCPGame(game, horizon, initial_conditions[1:end], context_state_block_dimensions)
    verbose || print("mcp game initialized\ninitializing mcp game solver ... ")
    m = MCPCoupledOptimizationSolver(mcp_game)
    verbose || println("mcp game solver initialized")
    return m
end

function get_dimensions(game::TrajectoryGame, horizon::Int)
    n_players = num_players(game)
    state_dims = map(1:n_players) do ii
        state_dim(game.dynamics.subsystems[ii])
    end
    x_size = sum(state_dims) * horizon
    player_xs = map(1:n_players) do ii
        mapreduce(vcat, 1:horizon) do t
            [(t-1)*sum(state_dims)+sum(state_dims[1:ii-1])+1 : (t-1)*sum(state_dims)+sum(state_dims[1:ii])]
        end
    end
    control_dims = map(1:n_players) do ii
        control_dim(game.dynamics.subsystems[ii])
    end
    u_size = sum(control_dims) * horizon
    player_us = map(1:n_players) do ii
        mapreduce(vcat, 1:horizon) do t
            [(t-1)*sum(control_dims)+sum(control_dims[1:ii-1])+1 : (t-1)*sum(control_dims)+sum(control_dims[1:ii])]
        end
    end
    (;n_players, state_dims, x_size, player_xs, control_dims, u_size, player_us)
end

function get_dimensions(game::MCPGame)
    get_dimensions(game.game, game.horizon)
end
