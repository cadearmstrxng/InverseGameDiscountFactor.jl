module WaymaxAnalysis

using CairoMakie
using BlockArrays

function eval_cc(
    states_file_m::String, contexts_file_m::String, actions_file_m::String,
    states_file_b::String, contexts_file_b::String, actions_file_b::String;
    output_file::String="cost_components.png")

    contexts_m = readlines(contexts_file_m)
    contexts_b = readlines(contexts_file_b)
    contexts_parsed_m = []
    contexts_parsed_b = []
    for (context_m, context_b) in zip(contexts_m, contexts_b)
        content_m = strip(context_m)
        if startswith(content_m, "[") && endswith(content_m, "]")
            content_m = content_m[2:end-1]
        end
        values_m = [parse(Float64, strip(val)) for val in split(content_m, ",") if !isempty(strip(val))]
        push!(contexts_parsed_m, values_m)
        
        content_b = strip(context_b)
        if startswith(content_b, "[") && endswith(content_b, "]")
            content_b = content_b[2:end-1]
        end
        values_b = [parse(Float64, strip(val)) for val in split(content_b, ",") if !isempty(strip(val))]
        push!(contexts_parsed_b, values_b)
    end
    contexts_m = [BlockArray(context, [9, 9, 9, 9]) for context in contexts_parsed_m]
    contexts_b = [BlockArray(context, [8, 8, 8, 8]) for context in contexts_parsed_b]

    states_m = readlines(states_file_m)
    states_b = readlines(states_file_b)
    states_parsed_m = []
    states_parsed_b = []
    for (state_m, state_b) in zip(states_m, states_b)
        content_m = strip(state_m)
        if startswith(content_m, "[") && endswith(content_m, "]")
            content_m = content_m[2:end-1]
        end
        values_m = [parse(Float64, strip(val)) for val in split(content_m, ",") if !isempty(strip(val))]
        push!(states_parsed_m, values_m)
        
        content_b = strip(state_b)
        if startswith(content_b, "[") && endswith(content_b, "]")
            content_b = content_b[2:end-1]
        end
        values_b = [parse(Float64, strip(val)) for val in split(content_b, ",") if !isempty(strip(val))]
        push!(states_parsed_b, values_b)
    end
    states_m = [BlockArray(state, [4, 4, 4, 4]) for state in states_parsed_m]
    states_b = [BlockArray(state, [4, 4, 4, 4]) for state in states_parsed_b]

    # actions_m = readlines(actions_file_m)
    # actions_b = readlines(actions_file_b)
    # actions_parsed_m = []
    # actions_parsed_b = []
    # for (action_m, action_b) in zip(actions_m, actions_b)
    #     content_m = strip(action_m)
    #     if startswith(content_m, "[") && endswith(content_m, "]")
    #         content_m = content_m[2:end-1]
    #     end
    #     values_m = [parse(Float64, strip(val)) for val in split(content_m, ",") if !isempty(strip(val))]
    #     push!(actions_parsed_m, values_m)
        
    #     content_b = strip(action_b)
    #     if startswith(content_b, "[") && endswith(content_b, "]")
    #         content_b = content_b[2:end-1]
    #     end
    #     values_b = [parse(Float64, strip(val)) for val in split(content_b, ",") if !isempty(strip(val))]
    #     push!(actions_parsed_b, values_b)
    # end
    # actions_m = [BlockArray(action, [2, 2, 2, 2]) for action in actions_parsed_m]
    # actions_b = [BlockArray(action, [2, 2, 2, 2]) for action in actions_parsed_b]

    num_players = 4
    horizon = length(states_m) - 12
    myopic = true 
    action_dim = 2

    function target_cost(x, context_state, t)
        norm_sqr(x[1:2] - context_state[1:2])
    end
    function control_cost(u, context_state, t)
        norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
    end
    function collision_cost(x, i, context_state, t)
        mapreduce(+, [1:(i-1); (i+1):num_players]) do paired_player
            -1 * norm_sqr(x[Block(i)][1:2] - x[Block(paired_player)][1:2])
        end
    end
    function lane_center_cost(x, i, context_state, t)
        (x[Block(i)][1] - 2555)^2
    end
    function road_boundary_cost(x, i, context_state, t)
        (x[Block(i)][1] - 2552)^2
    end
    function velocity_cost(x, i, context_state, t)
        norm_sqr(x[Block(i)][3] - 15)
    end

    cost_components = Dict{Symbol, Matrix{Float64}}()
    cost_components[:target] = zeros(horizon, num_players)
    # cost_components[:control] = zeros(horizon, num_players)
    cost_components[:collision] = zeros(horizon, num_players)
    cost_components[:lane_center] = zeros(horizon, num_players)
    cost_components[:road_boundary] = zeros(horizon, num_players)
    cost_components[:velocity] = zeros(horizon, num_players)

    for t in 1:horizon
        x_m = states_m[t+11]
        x_b = states_b[t+11]
        context_m = contexts_m[t]
        context_b = contexts_b[t]
        # u_m = actions_m[t]
        # u_b = actions_b[t]
        cost_components[:target][t, 1] = target_cost(x_m[Block(1)], context_m[Block(1)], t)
        # cost_components[:control][t, 1] = control_cost(u_m[Block(1)], context_m[Block(1)], t)
        cost_components[:collision][t, 1] = collision_cost(x_m, 1, context_m, t)
        cost_components[:lane_center][t, 1] = lane_center_cost(x_m, 1, context_m, t)
        cost_components[:road_boundary][t, 1] = road_boundary_cost(x_m, 1, context_m, t)
        cost_components[:velocity][t, 1] = velocity_cost(x_m, 1, context_m, t)

        cost_components[:target][t, 2] = target_cost(x_b[Block(1)], context_b[Block(1)], t)
        # cost_components[:control][t, 2] = control_cost(u_b[Block(1)], context_b[Block(1)], t)
        cost_components[:collision][t, 2] = collision_cost(x_b, 1, context_b, t)
        cost_components[:lane_center][t, 2] = lane_center_cost(x_b, 1, context_b, t)
        cost_components[:road_boundary][t, 2] = road_boundary_cost(x_b, 1, context_b, t)
        cost_components[:velocity][t, 2] = velocity_cost(x_b, 1, context_b, t)
    end

    CairoMakie.activate!()
    fig = Figure(resolution=(1200, 1600))
    # cost_names = ["target", "control", "collision", "lane_center", "road_boundary", "velocity"]
    cost_names = ["target", "collision", "lane_center", "road_boundary", "velocity"]
    for (idx, cname) in enumerate(cost_names)
        ax = Axis(fig[idx, 1], title=cname, xlabel="Timestep", ylabel="Cost")
        lines!(ax, 1:horizon, cost_components[Symbol(cname)][:, 1], label="Myopic")
        lines!(ax, 1:horizon, cost_components[Symbol(cname)][:, 2], label="Baseline")
        Legend(fig[idx, 2], ax)
    end
    save(output_file, fig)
    println("Saved cost component plot to $(output_file)")
    display(fig)
end
end