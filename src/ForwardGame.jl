module ForwardGame

import JuMP
import PATHSolver
import ..DynacmicsModelInterface
import ..JuMPUtils
import ..ForwardOptimalControl

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_game

function solve_game(
    control_system,
    player_cost_models,
    xₒ,
    T;
    solver = PATHSolver.Optimizer,
    solver_attributes = (; print_level = 3),
    init = (),
    match_equilibrium = nothing,
    verbose = false,
)

    n_players = length(player_cost_models)
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # decision variables
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T-1), 1:n_players])

    # Initialization
    JuMPUtils.init_if_hasproperty!(x, init, :x)
    JuMPUtils.init_if_hasproperty!(u, init, :u)
    JuMPUtils.init_if_hasproperty!(λ, init, :λ)

    # constraints
    @constraint(opt_model, x[:, 1] .== xₒ)
    DynacmicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)
    df = DynacmicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)

    for (player_idx, cost_model) in enumerate(player_cost_models)
        @unpack player_inputs, weights = cost_model
        dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)

        # KKT Nash Constraints
        @constraint(opt_model, 
            [t = 2:(T-1)], 
            dJ.dx[:,t] + λ[:, t-1, player_idx] - (λ[:, t, player_idx]' * df.dx[:, :, t])' .== 0
        )
        @constraint(
            opt_model,
            dJ.dx[:,T] + λ[:, T-1, player_idx] .== 0
        )

        @constraint(
            opt_model,
            [t = 1:T-1],
            dJ.du[:,t] - (λ[:, t, player_idx]' * df.du[:, player_inputs, t])' .== 0
        )
        @constraint(
            opt_model,
            dJ.du[player_inputs,T] .== 0
        )
    end

    if !isnothing(match_equilibrium)
        @objective(
            opt_model,
            Min,
            sum(el -> el^2, x - match_equilibrium.x)
        )
    end

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    JuMPUtils.isconverged(opt_model), JuMPUtils.get_values(; x, u, λ), opt_model
end

end












end