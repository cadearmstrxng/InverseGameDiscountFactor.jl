module InverseOptimalControl

import ..DynamicsModelInterface
import ..JuMPUtils
import ..CostUtils
import PATHSolver
import JuMP

using JuMP: @variable, @constraint, @objective
using UnPack: @unpack

export solve_inverse_optimal_control

#===============================================================================================#

function solve_inverse_optimal_control(
    y;
    control_system,
    cost_model,
    observation_model,
    fixed_inputs = (),
    init = (),
    solver = PATHSolver.Optimizer,
    solver_attributes = (; print_level = 3),
    cmin = 1e-5,
    max_observation_error = nothing,
    verbose = false,
    init_with_observation = true,
)

    T = size(y)[2]
    @unpack n_states, n_controls = control_system

    opt_model = JuMP.Model(solver)
    JuMPUtils.set_solver_attributes!(opt_model; solver_attributes...)

    # decision variables
    weights = @variable(opt_model, [keys(cost_model.weights)],)
    x = @variable(opt_model, [1:n_states, 1:T])
    u = @variable(opt_model, [1:n_controls, 1:T])
    λ = @variable(opt_model, [1:n_states, 1:(T-1)]) # multipliers of the forward optimality condition

    # Initialization
    if hasproperty(init, :weights) && !isnothing(init.weights)
        verbose && @info "Using weight guess: $(init.weights)"
        for k in keys(init.weights)
            JuMP.set_start_value(weights[k], init.weights[k])
        end
    else
        verbose && @info "Using weight default"
        JuMP.set_start_value.(weights, 1/length(weights))
    end

    if init_with_observation
        JuMP.set_start_value.(observation_model.expected_observation(x), y)
    end
    JuMPUtils.init_if_hasproperty!(u, init, :u)

    # constraints
    if iszero(observation_model.σ)
        @constraint(opt_model, observation_model.expected_observation(x[:,1]) .= y[:,1])
    end
    DynamicsModelInterface.add_dynamics_constraints!(control_system, opt_model, x, u)

    controlled_inputs = filter(i -> i ∉ fixed_inputs, 1:n_controls)
    for i in fixed_inputs
        JuMP.fix.(u[i, :], init.u[i, :])
    end

    df = DynamicsModelInterface.add_dynamics_jacobians!(control_system, opt_model, x, u)
    dJ = cost_model.add_objective_gradients!(opt_model, x, u; weights)
    @constraint(
        opt_model,
        [t = 2:(T-1)],
        dJ.dx[:,t] + λ[:, t-1] - (λ[:,t]' * df.dx[:, :, t])' .== 0
    )
    @constraint(
        opt_model,
        dJ.dx[:,T] + λ[:, T-1] .== 0
    )
    @constraint(
        opt_model,
        [t = 1:T-1],
        dJ.du[controlled_inputs,t] - (λ[:, t]' * df.du[:, controlled_inputs, t])' .== 0
    )
    @constraint(
        opt_model,
        dJ.du[controlled_inputs,T] .== 0
    )

    # regularization
    @constraint(
        opt_model,
        sum(weights) .== 1
    )
    @constraint(
        opt_model,
        weights .>= cmin
    )

    y_expected = observation_model.expected_observation(x)

    if !isnothing(max_observation_error)
        @constraint(
            opt_model,
            (y_expected - y) .^ 2 .<= max_observation_error^2
        )
    end

    # Objective: match the observed demonstration
    @objective(
        opt_model,
        Min,
        sum(el -> el^2, y_expected - y)
    )

    time = @elapsed JuMP.optimize!(opt_model)
    verbose && @info time

    solution = merge(
        CostUtils.namedtuple(JuMPUtils.get_values(; weights)),
        JuMPUtils.get_values(; x, u, λ),
    )

    JuMPUtils.isconverged(opt_model), solution, opt_model
end

end