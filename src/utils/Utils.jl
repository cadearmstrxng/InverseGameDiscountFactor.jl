function sample_initial_states_and_context(game, horizon, rng, collision_radius; 
    x0_range = 0.5, max_velocity = 0.5, myopic = true)
    println("Sampling initial states and goals...")

    game = game.game
    num_player = num_players(game)

    function sample_initial_state() # generic for both unicycle and bicycle dynamics
        [rand(rng) * x0_range, rand(rng) * x0_range, rand(rng) * max_velocity, rand(rng) * max_velocity] 
    end
    
    initial_states = Vector{Float64}[]
    context_states = Vector{Float64}[]
    for ii in 1:num_player
        initial_state = sample_initial_state()
        while !initial_state_feasibility(game, horizon, initial_state, initial_states, collision_radius)
            initial_state = sample_initial_state()
        end
        push!(initial_states, initial_state)
        if myopic
            context_state = [rand(rng) * 2, rand(rng) * 2, rand(rng)]
        else
            context_state = [rand(rng) * 2, rand(rng) * 2]
        end
        push!(context_states, context_state)
    end
    initial_states = mortar(initial_states)
    context_states = mortar(context_states)
    initial_states, context_states
end

function initial_state_feasibility(game, horizon, initial_state, initial_states, collision_radius)

    if length(initial_states) == 0
        return true
    else
        control_dimension = control_dim(game.dynamics.subsystems[1])
        dummy_strategy = (x, t) -> zeros(control_dimension)
        collision_detection_steps = 5

        for other_state in initial_states
            other_rollout = rollout(game.dynamics.subsystems[1], 
                dummy_strategy, other_state, horizon)
            this_rollout = rollout(game.dynamics.subsystems[1], 
                dummy_strategy, initial_state, horizon)
            for tt in 1:collision_detection_steps
                if norm(this_rollout.xs[tt][1:2] - other_rollout.xs[tt][1:2]) < 2.05 * collision_radius
                    return false
                end
            end
        end
    end
    return true
end

"""
Structure of output is as follows:

 - [[x₁¹; x₂¹];[x₁²; x₂²];...;[x₁ᵀ; x₂ᵀ]] where xᵢʲ is the state of player i at time j in block vectors for each time

This structure is not consistent with MyopicSolver, do not pass in result to myopic solver.

"""
function reconstruct_solution(solution, game, horizon)
    num_player = num_players(game)
    player_state_dimension = convert(Int64, state_dim(game.dynamics)/num_player)

    if typeof(solution) == NamedTuple{(:primals, :variables, :status), Tuple{Vector{Vector{ForwardDiff.Dual{Nothing, Float64, 14}}}, Vector{Float64}, PATHSolver.MCP_Termination}} || typeof(solution) == NamedTuple{(:primals, :variables, :status), Tuple{Vector{Vector{ForwardDiff.Dual{Nothing, Float64, 12}}}, Vector{Float64}, PATHSolver.MCP_Termination}}
        primals = solution.primals
        solution = []
        for primal in primals
            vars = []
            for i in primal
                push!(vars, i.value)
            end
            push!(solution,vars)
        end
        player1state = solution[1][1:player_state_dimension*horizon]
        player2state = solution[2][1:player_state_dimension*horizon]
    else
        player1state = solution.primals[1][1:player_state_dimension*horizon]
        player2state = solution.primals[2][1:player_state_dimension*horizon]
    end
    solution = []
    for i in 1:horizon
        push!(solution, player1state[(i-1) * player_state_dimension + 1: i * player_state_dimension])
        push!(solution, player2state[(i-1) * player_state_dimension + 1: i * player_state_dimension])
    end
    solution = vcat(solution...)
    solution = BlockVector(solution, [2*player_state_dimension for i in 1:horizon])
end
