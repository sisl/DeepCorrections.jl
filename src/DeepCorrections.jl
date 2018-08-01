module DeepCorrections

using POMDPs
using TensorFlow
using DeepRL
using DeepQLearning
import DeepQLearning.build_graph, DeepQLearning.dqn_train

export 
    DeepCorrectionSolver,
    lowfi_values,
    zeros_values,
    correction,
    additive_correction,
    multiplicative_correction

include("solver.jl")

end # module
