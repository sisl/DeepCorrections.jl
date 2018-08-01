"""
Deep Correction algorithm to train an additive correction term to an existing value function 
    dqn::DeepQLearningSolver
    lowfi_values::Any 
Function or object for the low fidelity value function to correct, returns a vector of action values, ordered according to action_index
If this is a function `f`, `f(mdp, s)` will be called to estimate the value.
If this is an object `o`, `lowfi_values(o, mdp, s, depth)` will be called.
default: zero_values::Function
    correction::Any Function or object for the correction method used 
If this is a function `f`, `f(mdp, q_lo, q_corr, weight)` will be called to estimate the value.
If this is an object `o`, `correction(o, mdp, q_lo, q_corr, weight)` will be called.
default: additive_correction::Function
    correction_weight::Float64 a constant weight that provides more flexibility in tuning the correction method
""" 
mutable struct DeepCorrectionSolver <: Solver
    dqn::DeepQLearningSolver
    lowfi_values::Any
    correction::Any
    correction_weight::Float64
end

function DeepCorrectionSolver(;correction_network::QNetworkArchitecture = QNetworkArchitecture(fc=[16, 8]),
                              lowfi_values::Any = zeros_values, 
                              correction_method::Any = additive_correction,
                              correction_weight::Float64 = 1.0,
                              lr::Float64 = 0.005,
                              max_steps::Int64 = 1000,
                              target_update_freq::Int64 = 500,
                              batch_size::Int64 = 32,
                              train_freq::Int64  = 4,
                              log_freq::Int64 = 100,
                              eval_freq::Int64 = 100,
                              num_ep_eval::Int64 = 100,
                              eps_fraction::Float64 = 0.5,
                              eps_end::Float64 = 0.01,
                              double_q::Bool = true,
                              dueling::Bool = true,
                              prioritized_replay::Bool = true,
                              prioritized_replay_alpha::Float64 = 0.6,
                              prioritized_replay_epsilon::Float64 = 1e-6,
                              prioritized_replay_beta::Float64 = 0.4,
                              buffer_size::Int64 = 1000,
                              max_episode_length::Int64 = 100,
                              train_start::Int64 = 200,
                              grad_clip::Bool = true,
                              clip_val::Float64 = 10.0,
                              rng::AbstractRNG = MersenneTwister(0),
                              logdir::String = "log",
                              save_freq::Int64 = 10000,
                              evaluation_policy::Any = basic_evaluation,
                              exploration_policy::Any = linear_epsilon_greedy(max_steps, eps_fraction, eps_end),
                              verbose::Bool = true)
    dqn = DeepQLearningSolver(correction_network, lr, max_steps, target_update_freq, batch_size, train_freq, log_freq,
                              eval_freq, num_ep_eval, eps_fraction, eps_end, double_q, dueling, prioritized_replay,
                              prioritized_replay_alpha, prioritized_replay_epsilon, prioritized_replay_beta,
                              buffer_size, max_episode_length, train_start, grad_clip, clip_val, rng, logdir,
                              save_freq, evaluation_policy, exploration_policy, verbose)
    return DeepCorrectionSolver(dqn, lowfi_values, correction_method, correction_weight)
end


struct DeepCorrectionPolicy <: AbstractNNPolicy
    q::Tensor # Q network
    s::Tensor # placeholder
    lowfi_values::Any
    correction::Any
    correction_weight::Float64
    env::AbstractEnvironment
    sess
end

"""
    lowfi_values 
return a low fidelity estimate of the action values, the returned value should be a vector of size `n_actions(mdp)``
the actions are ordered according to `action_index`
"""
function lowfi_values end
lowfi_values(f::Function, problem::Union{POMDP,MDP}, s) = f(problem, s)


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
correction(f::Function, problem::Union{POMDP, MDP}, q_lo::Array{Float64}, q_corr::Array{Float64}, weight::Float64) = f(problem, q_lo, q_corr, weight)

# default implementation 
function additive_correction(problem::Union{POMDP, MDP}, q_lo::Array{Float64}, q_corr::Array{Float64}, weight::Float64)
    return q_lo + q_corr
end

function multiplicative_correction(problem::Union{POMDP, MDP}, q_lo::Array{Float64}, q_corr::Array{Float64}, weight::Float64)
    return q_lo.*q_corr
end


function POMDPs.solve(solver::DeepCorrectionSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.dqn.rng)
    #init session and build graph Create a TrainGraph object with all the tensors
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepCorrectionSolver, env::AbstractEnvironment)
    train_graph = build_graph(solver.dqn, env)

    # init and populate replay buffer
    if solver.dqn.prioritized_replay
        replay = PrioritizedReplayBuffer(env, solver.dqn.buffer_size, solver.dqn.batch_size)
    else
        replay = ReplayBuffer(env, solver.dqn.buffer_size, solver.dqn.batch_size)
    end
    populate_replay_buffer!(replay, env, max_pop=solver.dqn.train_start)
    # init variables
    run(train_graph.sess, global_variables_initializer())
    # train model
    policy = DeepCorrectionPolicy(train_graph.q, train_graph.s, solver.lowfi_values, solver.correction, solver.correction_weight,
                                  env, train_graph.sess)
    dqn_train(solver.dqn, env, train_graph, policy, replay)
    return policy
end

function DeepQLearning.get_action(policy::DeepCorrectionPolicy, o::Array{Float64})
    # cannot take a batch of observations
    q_low = lowfi_values(policy.lowfi_values, policy.env.problem, o)
    q_low = reshape(q_low, (1, length(q_low)))
    o_batch = reshape(o, (1, size(o)...))
    TensorFlow.set_def_graph(policy.sess.graph)
    q_tf = run(policy.sess, policy.q, Dict(policy.s => o_batch))
    q_corr = convert(Array{Float64}, q_tf)
    q_val = correction(policy.correction, policy.env.problem, q_low, q_corr, policy.correction_weight)
    ai = indmax(q_val)
    return actions(policy.env)[ai]
end

