include("../graphing/ExperimentGraphingUtils.jl")
using .ExperimentGraphingUtils

output_dir = "experiments/crosswalk/results"
isdir(output_dir) || mkpath(output_dir)

process_and_graph_crosswalk_results(output_prefix=output_dir) 