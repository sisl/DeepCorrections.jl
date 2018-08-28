function POMDPs.solve(solver::DeepCorrectionSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.dqn.rng)
    #init session and build graph Create a TrainGraph object with all the tensors
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepCorrectionSolver, env::AbstractEnvironment)
    train_graph = build_graph(solver, env)

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

function DeepQLearning.get_action!(policy::DeepCorrectionPolicy, o::Array{Float64})
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

function DeepQLearning.get_action(policy::DeepCorrectionPolicy, o::Array{Float64})
    return get_action!(policy, o)
end

function DeepQLearning.reset_hidden_state!(policy::DeepCorrectionPolicy)
    # no hidden states
end

function batch_train!(env::AbstractEnvironment, graph::CorrectionTrainGraph, replay::ReplayBuffer)
    weights = ones(replay.batch_size)
    s_batch, a_batch, r_batch, sp_batch, done_batch = sample(replay)
    q_lo_batch, q_lo_p_batch = get_q_lo_batch(graph, env, s_batch, sp_batch)
    return batch_train!(graph, s_batch, a_batch, r_batch, sp_batch, done_batch, weights, q_lo_batch, q_lo_p_batch)
end

function batch_train!(env::AbstractEnvironment, graph::CorrectionTrainGraph, replay::PrioritizedReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
    q_lo_batch, q_lo_p_batch = get_q_lo_batch(graph, env, s_batch, sp_batch)
    return batch_train!(graph, s_batch, a_batch, r_batch, sp_batch, done_batch, weights, q_lo_batch, q_lo_p_batch)
end

function get_q_lo_batch(graph::CorrectionTrainGraph, env::AbstractEnvironment, s_batch, sp_batch)
    bs = size(s_batch, 1)
    q_lo_batch = zeros(bs, n_actions(env))
    q_lo_p_batch = zeros(bs, n_actions(env))
    for i=1:bs
        q_lo_batch[i, :] = lowfi_values(graph.lowfi_values, env.problem, reshape(s_batch, bs, :)[i, :])
        q_lo_p_batch[i, :] = lowfi_values(graph.lowfi_values, env.problem, reshape(sp_batch, bs, :)[i, :])
    end
    return q_lo_batch, q_lo_p_batch 
end

function batch_train!(graph::CorrectionTrainGraph, s_batch, a_batch, r_batch, sp_batch, done_batch, weights, q_lo_batch, q_lo_p_batch)
    tf.set_def_graph(graph.sess.graph)
    feed_dict = Dict(graph.s => s_batch,
                    graph.a => a_batch,
                    graph.sp => sp_batch,
                    graph.r => r_batch,
                    graph.done_mask => done_batch,
                    graph.importance_weights => weights,
                    graph.q_lo => q_lo_batch,
                    graph.q_lo_p => q_lo_p_batch)
    loss_val, td_errors, grad_val, _ = run(graph.sess,[graph.loss, graph.td_errors, graph.grad_norm, graph.train_op],
                                feed_dict)
    return (loss_val, td_errors, grad_val)
end