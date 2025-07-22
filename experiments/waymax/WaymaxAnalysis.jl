module WaymaxAnalysis

using CairoMakie
using BlockArrays
using Statistics
using LinearAlgebra

function parse_data_file(file_path::String)
    lines = readlines(file_path)
    parsed_data = []
    for line in lines
        content = strip(line)
        if startswith(content, "[") && endswith(content, "]")
            content = content[2:end-1]
        end
        values = [parse(Float64, strip(val)) for val in split(content, ",") if !isempty(strip(val))]
        push!(parsed_data, values)
    end
    return parsed_data
end

function eval_cc(; input_dir::String="./experiments/waymax/mc")
    myopic_dir = joinpath(input_dir, "myopic")
    baseline_dir = joinpath(input_dir, "baseline")
    output_dir = joinpath(input_dir, "graphs")
    mkpath(output_dir)

    cost_names = ["target", "collision", "lane_center", "road_boundary", "velocity"]
    
    cost_results = Dict{Symbol, Dict{Float64, Dict{String, Vector{Float64}}}}()
    for cname in cost_names
        cost_results[Symbol(cname)] = Dict{Float64, Dict{String, Vector{Float64}}}()
    end

    discount_results = Dict{Float64, Dict{Int, Vector{Float64}}}()

    file_pattern = r"states_([0-9\.]+)@([0-9]+)\.txt"

    for filename in readdir(myopic_dir)
        m = match(file_pattern, filename)
        if m === nothing
            continue
        end
        
        noise_level_str, trial_num_str = m.captures
        noise_level = parse(Float64, noise_level_str)
        trial_num = parse(Int, trial_num_str)
        # println("Processing noise $(noise_level), trial $(trial_num)")

        states_file_m = joinpath(myopic_dir, "states_$(noise_level)@$(trial_num).txt")
        states_file_b = joinpath(baseline_dir, "states_$(noise_level)@$(trial_num).txt")
        contexts_file_m = joinpath(myopic_dir, "contexts_$(noise_level)@$(trial_num).txt")
        contexts_file_b = joinpath(baseline_dir, "contexts_$(noise_level)@$(trial_num).txt")

        if !isfile(states_file_b) || !isfile(contexts_file_b)
            println("Skipping, missing baseline file for noise $(noise_level), trial $(trial_num)")
            continue
        end
        
        contexts_parsed_m = parse_data_file(contexts_file_m)
        contexts_parsed_b = parse_data_file(contexts_file_b)
        contexts_m = [BlockArray(context, [9, 9, 9, 9]) for context in contexts_parsed_m]
        contexts_b = [BlockArray(context, [8, 8, 8, 8]) for context in contexts_parsed_b]

        states_parsed_m = parse_data_file(states_file_m)
        states_parsed_b = parse_data_file(states_file_b)
        states_m = [BlockArray(state, [4, 4, 4, 4]) for state in states_parsed_m]
        states_b = [BlockArray(state, [4, 4, 4, 4]) for state in states_parsed_b]

        num_players = 4
        horizon = length(states_m) - 12
        myopic = true 

        if !haskey(discount_results, noise_level)
            discount_results[noise_level] = Dict(i => Float64[] for i in 1:num_players)
        end
        
        for t in 1:horizon
            context_m = contexts_m[t]
            for p in 1:num_players
                discount_factor = context_m[Block(p)][3]
                push!(discount_results[noise_level][p], discount_factor)
            end
        end

        function target_cost(x, context_state, t)
            norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u, context_state, t)
            norm_sqr(u) * (myopic ? context_state[3] ^ t : 1)
        end
        function collision_cost(x, i, context_state, t)
            min(map([1:(i-1); (i+1):num_players]) do paired_player
                norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2])
            end...)
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
        for cname in cost_names
            cost_components[Symbol(cname)] = zeros(horizon, 2)
        end
        
        for t in 1:horizon
            x_m = states_m[t+11]
            x_b = states_b[t+11]
            context_m = contexts_m[t]
            context_b = contexts_b[t]
            cost_components[:target][t, 1] = target_cost(x_m[Block(1)], context_m[Block(1)], t)
            cost_components[:collision][t, 1] = collision_cost(x_m, 1, context_m, t)
            cost_components[:lane_center][t, 1] = lane_center_cost(x_m, 1, context_m, t)
            cost_components[:road_boundary][t, 1] = road_boundary_cost(x_m, 1, context_m, t)
            cost_components[:velocity][t, 1] = velocity_cost(x_m, 1, context_m, t)

            cost_components[:target][t, 2] = target_cost(x_b[Block(1)], context_b[Block(1)], t)
            cost_components[:collision][t, 2] = collision_cost(x_b, 1, context_b, t)
            cost_components[:lane_center][t, 2] = lane_center_cost(x_b, 1, context_b, t)
            cost_components[:road_boundary][t, 2] = road_boundary_cost(x_b, 1, context_b, t)
            cost_components[:velocity][t, 2] = velocity_cost(x_b, 1, context_b, t)
        end
        
        for cname in cost_names
            cname_sym = Symbol(cname)
            if !haskey(cost_results[cname_sym], noise_level)
                cost_results[cname_sym][noise_level] = Dict("Myopic" => Float64[], "Baseline" => Float64[])
            end

            if cname_sym == :collision
                push!(cost_results[cname_sym][noise_level]["Myopic"], minimum(cost_components[cname_sym][:, 1]))
                push!(cost_results[cname_sym][noise_level]["Baseline"], minimum(cost_components[cname_sym][:, 2]))
            else
                append!(cost_results[cname_sym][noise_level]["Myopic"], cost_components[cname_sym][:, 1])
                append!(cost_results[cname_sym][noise_level]["Baseline"], cost_components[cname_sym][:, 2])
            end
        end
    end
    
    noise_levels = sort(collect(keys(cost_results[first(keys(cost_results))])))
    x_positions = 1:length(noise_levels)

    for cname in cost_names
        cname_sym = Symbol(cname)
        fig = Figure(resolution=(1200, 400), fontsize=20)
        title_label = cname_sym == :collision ? "Distance to Collision" : titlecase(cname)
        ax = Axis(fig[1, 1], title=title_label, xlabel="Noise σ [m]", ylabel="Cost [m]", 
                  xticks=(x_positions, string.(noise_levels)),
                  xticklabelrotation=π/4)

        for (i, nl) in enumerate(noise_levels)
            myopic_data = cost_results[cname_sym][nl]["Myopic"]
            baseline_data = cost_results[cname_sym][nl]["Baseline"]
            
            myopic_x_center = i - 0.15
            baseline_x_center = i + 0.15

            if !isempty(myopic_data)
                violin!(ax, fill(myopic_x_center, length(myopic_data)), myopic_data, width=0.3, color=(:blue, 0.5))
                med = median(myopic_data)
                lines!(ax, [myopic_x_center - 0.075, myopic_x_center + 0.075], [med, med], color=:black, linewidth=2)
            end

            if !isempty(baseline_data)
                violin!(ax, fill(baseline_x_center, length(baseline_data)), baseline_data, width=0.3, color=(:orange, 0.5))
                med = median(baseline_data)
                lines!(ax, [baseline_x_center - 0.075, baseline_x_center + 0.075], [med, med], color=:black, linewidth=2)
            end
        end

        elems = [
            PolyElement(color = :blue, strokecolor=:black), 
            PolyElement(color = :orange, strokecolor=:black),
            LineElement(color=:black, linewidth=2)
        ]
        labels = ["Myopic", "Baseline", "Median"]

        if cname_sym == :collision
            all_myopic_mins = vcat(get.(values(cost_results[cname_sym]), "Myopic", Ref([]))...)
            all_baseline_mins = vcat(get.(values(cost_results[cname_sym]), "Baseline", Ref([]))...)
            
            global_min_myopic = isempty(all_myopic_mins) ? Inf : minimum(all_myopic_mins)
            global_min_baseline = isempty(all_baseline_mins) ? Inf : minimum(all_baseline_mins)

            if isfinite(global_min_myopic)
                hlines!(ax, global_min_myopic, color=:cyan, linestyle=:dash, linewidth=2)
                push!(elems, LineElement(color=:cyan, linestyle=:dash, linewidth=2))
                push!(labels, "Overall Min (Myopic): $(round(global_min_myopic, digits=2))")
            end
            if isfinite(global_min_baseline)
                hlines!(ax, global_min_baseline, color=:magenta, linestyle=:dash, linewidth=2)
                push!(elems, LineElement(color=:magenta, linestyle=:dash, linewidth=2))
                push!(labels, "Overall Min (Baseline): $(round(global_min_baseline, digits=2))")
            end
        end

        Legend(fig[1, 2], elems, labels, "Models")
        
        output_file = joinpath(output_dir, "$(cname)_graph.pdf")
        save(output_file, fig)
        println("Saved violin plot to $(output_file)")
    end

    # Plot for discount factors
    fig_df = Figure(resolution=(1200, 800))
    ax_df = Axis(fig_df[1, 1], title="Recovered Discount Factors by Player", xlabel="Noise Level", ylabel="Discount Factor", 
                 xticks=(x_positions, string.(noise_levels)),
                 xticklabelrotation=π/4)

    num_players = 4
    players_to_plot = 2:num_players
    num_plotted = length(players_to_plot)
    colors = [:blue, :orange, :green, :red]
    violin_width = 0.2

    for (i, nl) in enumerate(noise_levels)
        if !haskey(discount_results, nl)
            continue
        end
        for (idx, p) in enumerate(players_to_plot)
            player_data = discount_results[nl][p]
            if isempty(player_data)
                continue
            end
            offset = (idx - (num_plotted + 1) / 2) * violin_width
            x_center = i + offset
            violin!(ax_df, fill(x_center, length(player_data)), player_data, width=violin_width, color=(colors[p], 0.5), bandwidth=0.0001)
            med = median(player_data)
            line_width = violin_width * 0.4
            lines!(ax_df, [x_center - line_width/2, x_center + line_width/2], [med, med], color=:black, linewidth=2)
        end
    end

    elems_df = Any[PolyElement(color=colors[p], strokecolor=:black) for p in players_to_plot]
    labels_df = ["Player $p" for p in players_to_plot]
    push!(elems_df, LineElement(color=:black, linewidth=2))
    push!(labels_df, "Median")
    Legend(fig_df[1, 2], elems_df, labels_df, "Players")
    
    output_file_df = joinpath(output_dir, "discount_factors_graph.pdf")
    save(output_file_df, fig_df)
    println("Saved discount factor graph to $(output_file_df)")
end

function norm_sqr(x::Vector{Float64})
    sum(x.^2)
end
function norm_sqr(x::Float64)
    x^2
end

function eval_2(; input_dir::String="./experiments/waymax/mc")
    myopic_dir = joinpath(input_dir, "myopic")
    output_dir = joinpath(input_dir, "graphs")
    mkpath(output_dir)

    # --- Pre-scan to get dimensions ---
    file_pattern = r"states_([0-9\.]+)@([0-9]+)\.txt"
    all_files = readdir(myopic_dir)
    matching_files = [f for f in all_files if match(file_pattern, f) !== nothing]

    if isempty(matching_files)
        println("No data files found in $myopic_dir")
        return
    end

    all_trials = sort(unique([parse(Int, match(file_pattern, f).captures[2]) for f in matching_files]))
    all_noise_levels = sort(unique([parse(Float64, match(file_pattern, f).captures[1]) for f in matching_files]))
    num_trials = length(all_trials)
    trial_map = Dict(trial_num => i for (i, trial_num) in enumerate(all_trials))

    m = match(file_pattern, matching_files[1])
    noise_level_str, trial_num_str = m.captures
    states_file_sample = joinpath(myopic_dir, "states_$(noise_level_str)@$(trial_num_str).txt")
    states_parsed_sample = parse_data_file(states_file_sample)
    horizon = length(states_parsed_sample) - 12
    num_players = 4

    # --- Initialize data stores ---
    collision_data = Dict(nl => Dict(p => fill(Inf, num_trials, horizon) for p in 2:num_players) for nl in all_noise_levels)
    target_data = Dict(nl => fill(Inf, num_trials, horizon) for nl in all_noise_levels)

    # --- Data Processing ---
    for filename in matching_files
        m = match(file_pattern, filename)
        noise_level = parse(Float64, m.captures[1])
        trial_num = parse(Int, m.captures[2])
        trial_idx = trial_map[trial_num]
        states_file_m = joinpath(myopic_dir, "states_$(noise_level)@$(trial_num).txt")
        contexts_file_m = joinpath(myopic_dir, "contexts_$(noise_level)@$(trial_num).txt")
        
        contexts_m = [BlockArray(context, [9, 9, 9, 9]) for context in parse_data_file(contexts_file_m)]
        states_m = [BlockArray(state, [4, 4, 4, 4]) for state in parse_data_file(states_file_m)]

        target_dist(x, context_state) = norm(x[1:2] - context_state[1:2])

        for t in 1:horizon
            x_m = states_m[t+11]
            context_m = contexts_m[t]
            
            target_data[noise_level][trial_idx, t] = target_dist(x_m[Block(1)], context_m[Block(1)])
            for p in 2:num_players
                collision_data[noise_level][p][trial_idx, t] = norm(x_m[Block(1)][1:2] - x_m[Block(p)][1:2])
            end
        end
    end

    # --- Plotting ---
    for nl in all_noise_levels
        fig = Figure(resolution=(1200, 400), fontsize=24)
        ax = Axis(fig[1, 1], xlabel="Time Step", ylabel="Distance [m]", title="Metrics for Noise Standard Deviation = $nl")

        time_steps = 0:horizon-1
        player_colors = Dict(2 => :orange, 3 => :green, 4 => :red)

        for p in 2:num_players
            for trial_idx in 1:num_trials
                series = collision_data[nl][p][trial_idx, :]
                lines!(ax, time_steps, series, color=(player_colors[p], 0.2))
            end
        end

        for trial_idx in 1:num_trials
            series = target_data[nl][trial_idx, :]
            lines!(ax, time_steps, series, color=(:black, 0.2))
        end

        legend_elems = [
            LineElement(color=:orange), LineElement(color=:green), LineElement(color=:red),
            LineElement(color=:black)
        ]
        legend_labels = [
            "ego-car #3 dist (safety)", "ego-car #8 dist (safety)", "ego-car #2 dist (safety)",
            "ego-goal dist (efficiency)"
        ]
        
        all_collision_distances = vcat(values(collision_data[nl])...)
        if !isempty(all_collision_distances)
            global_min_dist = minimum(all_collision_distances)
            if isfinite(global_min_dist)
                hlines!(ax, global_min_dist, color=:purple, linestyle=:dash, linewidth=2)
                push!(legend_elems, LineElement(color=:purple, linestyle=:dash))
                push!(legend_labels, "overall min dist: $(round(global_min_dist, digits=2)) [m]")
            end
        end

        Legend(fig[1, 2], legend_elems, legend_labels, "Metrics")

        output_file = joinpath(output_dir, "waymax_graph_noise_$(nl).pdf")
        save(output_file, fig)
        println("Saved time-series graph to $(output_file)")
    end
end
    
    
end