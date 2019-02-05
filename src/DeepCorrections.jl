module DeepCorrections

using LinearAlgebra
using POMDPs
using POMDPModelTools
using POMDPPolicies
using Flux
using RLInterface
using DeepQLearning
using DeepQLearning: isrecurrent, create_dueling_network, sample, globalnorm
using DeepQLearning: initialize_replay_buffer, update_priorities!
using DeepQLearning: batch_train!, dqn_train!
using Parameters

export 
    DeepCorrectionSolver,
    DeepCorrectionPolicy,
    CorrectionTrainGraph,
    lowfi_values,
    zeros_values,
    correction,
    additive_correction,
    multiplicative_correction

"""
Deep Correction algorithm to train an additive correction term to an existing value function 

# Fields
    - dqn::DeepQLearningSolver
See the documentation in DeepQLearning.jl for the field of the solver, the qnetwork corresponds to the correction network
    - lowfi_values::Any 
Function or object for the low fidelity value function to correct, returns a vector of action values, ordered according to action_index
If this is a function `f`, `f(mdp, s)` will be called to estimate the value.
If this is an object `o`, `lowfi_values(o, mdp, s, depth)` will be called.
default: zero_values::Function
    - correction::Any Function or object for the correction method used 
If this is a function `f`, `f(mdp, q_lo, q_corr, weight)` will be called to estimate the value.
If this is an object `o`, `correction(o, mdp, q_lo, q_corr, weight)` will be called.
default: additive_correction::Function
    - correction_weight::Float64 a constant weight that provides more flexibility in tuning the correction method
""" 
@with_kw mutable struct DeepCorrectionSolver <: Solver
    dqn::DeepQLearningSolver = DeepQLearningSolver()
    lowfi_values::Any = zeros_values
    correction::Any = additive_correction
    correction_weight::Float64 = 1.0
end

"""
    lowfi_values 
return a low fidelity estimate of the action values, the returned value should be a vector of size `n_actions(mdp)``
the actions are ordered according to `action_index`
"""
function lowfi_values end
lowfi_values(f::Function, problem::Union{POMDP,MDP}, s) = f(problem, s)
lowfi_values(p::Policy, problem::Union{POMDP, MDP}, s) = actionvalues(p, s)

# default implementation
# should return a vector of size n_actions
function zeros_values(problem, s) 
    na = n_actions(problem)
    return zeros(na)
end

"""
    correction
combine the output from the corrective network and the low fidelity values 
"""
function correction end 
correction(f::Function, problem::Union{POMDP, MDP}, q_lo::AbstractArray, q_corr::AbstractArray, weight::Float64) = f(problem, q_lo, q_corr, weight) 

# default implementations

function additive_correction(problem::Union{POMDP, MDP}, q_lo::AbstractArray, q_corr::AbstractArray, weight::Float64)
    return q_lo .+ weight.*q_corr
end

function multiplicative_correction(problem::Union{POMDP, MDP}, q_lo::AbstractArray, q_corr::AbstractArray, weight::Float64)
    return q_lo.*q_corr
end

include("policy.jl")
include("solver.jl")

end # module
