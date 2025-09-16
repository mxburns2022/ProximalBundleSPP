include("ProxBundle.jl")
using ArgParse




s = ArgParseSettings()
@add_arg_table s begin
    "--n"
    help = "Matrix row dimension"
    arg_type = Int
    required = true
    "--m"
    help = "Matrix column dimension"
    arg_type = Int
    required = true
    "--density"
    help = "Density for A matrix"
    arg_type = Float64
    default = 0.05
    "--penalty_x"
    help = "L∞ Norm Regularization (x)"
    arg_type = Float64
    default = 1.0
    "--penalty_y"
    help = "L∞ Norm Regularization (y)"
    arg_type = Float64
    default = 1.0
    "--epsilon"
    help = "Target accuracy"
    arg_type = Float64
    default = 1e-3
    "--numcuts"
    help = "Number of cuts"
    arg_type = Int
    default = 1
    "--seed"
    help = "Random seed"
    arg_type = Int
    default = 1
    "--frequency"
    help = "Iteration logging frequency"
    arg_type = Int
    default = 1000
    "--solver"
    help = "Solver method"
    arg_type = String
    default = "subgradient"
    "--dynamic-stepsize"
    help = "Use dynamic stepsize for subgradient method"
    action = :store_true
    "--stddev"
    help = "Value standard deviation for randomly generated A"
    arg_type = Float64
    default = 1.0
end
parsed_args = parse_args(ARGS, s)
game = generate_random_zero_sum(parsed_args["n"], parsed_args["m"], parsed_args["density"], parsed_args["stddev"]; seed=parsed_args["seed"], sparse=parsed_args["density"] < 0.1)
if parsed_args["solver"] == "bundle"
    bundle_saddle_point(game, parsed_args["epsilon"]; memory=parsed_args["numcuts"], log_frequency=parsed_args["frequency"])
else
    subgradient_method(game, parsed_args["epsilon"]; log_frequency=parsed_args["frequency"], dynamic_stepsize=parsed_args["dynamic-stepsize"])
end
