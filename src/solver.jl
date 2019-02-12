function POMDPs.solve(solver::DeepCorrectionSolver, problem::MDP)
    env = MDPEnvironment(problem, rng=solver.dqn.rng)
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepCorrectionSolver, problem::POMDP)
    env = POMDPEnvironment(problem, rng=solver.dqn.rng)
    return solve(solver, env)
end

function POMDPs.solve(solver::DeepCorrectionSolver, env::AbstractEnvironment)
    # check reccurence 
    if isrecurrent(solver.dqn.qnetwork) && !solver.dqn.recurrence
        throw("DeepQLearningError: you passed in a recurrent model but recurrence is set to false")
    end
    replay = initialize_replay_buffer(solver.dqn, env)
    if solver.dqn.dueling 
        active_q = create_dueling_network(solver.dqn.qnetwork)
        solver.dqn.qnetwork = active_q
    else
        active_q = solver.dqn.qnetwork
    end
    correction_network = NNPolicy(env.problem, active_q, ordered_actions(env.problem), length(obs_dimensions(env)))
    policy = DeepCorrectionPolicy(env.problem, correction_network, solver.lowfi_values, solver.correction, solver.correction_weight, ordered_actions(env.problem))
    return dqn_train!(solver.dqn, env, policy, replay)
end

function DeepQLearning.batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      policy::DeepCorrectionPolicy,
                      optimizer, 
                      target_q,
                      replay::PrioritizedReplayBuffer)
    s_batch, a_batch, r_batch, sp_batch, done_batch, indices, weights = sample(replay)
    loss_val, td_vals, grad_norm = batch_train!(solver, env, policy, optimizer, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, weights)
    update_priorities!(replay, indices, td_vals)
    return loss_val, td_vals, grad_norm
end

function DeepQLearning.batch_train!(solver::DeepQLearningSolver,
                      env::AbstractEnvironment,
                      policy::DeepCorrectionPolicy,
                      optimizer,
                      target_q,
                      s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights)
    active_q = solver.qnetwork
    loss_tracked, td_tracked = deep_correction_loss(solver, env, policy, active_q, target_q, s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights)
    loss_val = loss_tracked.data
    td_vals = Flux.data(td_tracked)
    p = params(active_q)
    Flux.back!(loss_tracked)
    grad_norm = globalnorm(p)
    Flux._update_params!(optimizer, p)
    return loss_val, td_vals, grad_norm
end

function deep_correction_loss(solver::DeepQLearningSolver, 
                              env::AbstractEnvironment, 
                              policy::DeepCorrectionPolicy,
                              active_q,
                              target_q,
                              s_batch, a_batch, r_batch, sp_batch, done_batch, importance_weights)
    q_corr = active_q(s_batch)
    q_corr_sa = diag(view(q_corr, a_batch, :))
    q_lo = get_qlo_batch(solver, policy, env, s_batch)
    q_lo_sa = diag(view(q_lo, a_batch, :))
    q_sa = correction(policy.correction, env.problem, q_lo_sa, q_corr_sa, policy.correction_weight)
    q_lo_p = get_qlo_batch(solver, policy, env, sp_batch)
    if solver.double_q
        target_q_corr = target_q(sp_batch)
        q_sp = correction(policy.correction, env.problem, q_lo_p, target_q_corr, policy.correction_weight)
        qp_corr_p = active_q(sp_batch)
        qp_values = correction(policy.correction, env.problem, q_lo_p, qp_corr_p, policy.correction_weight)
        best_a = Flux.onecold(qp_values)
        q_sp_max = diag(view(q_sp, best_a, :))
    else
        target_q_corr = target_q(sp_batch)
        q_sp = correction(policy.correction, env.problem, q_lo_p, target_q_corr, policy.correction_weight)
        q_sp_max = @view maximum(q_sp, dims=1)[:]
    end
    q_targets = r_batch .+ convert(Vector{Float32}, (1.0 .- done_batch).*discount(env.problem)).*q_sp_max 
    td_tracked = q_sa .- q_targets
    loss_tracked = mean(huber_loss, importance_weights.*td_tracked)
    return loss_tracked, td_tracked
end 

function get_qlo_batch(solver::DeepQLearningSolver, policy::DeepCorrectionPolicy, env::AbstractEnvironment, s_batch::AbstractArray)
    q_lo = zeros(Float32, n_actions(env), solver.batch_size)
    for i=1:solver.batch_size
        q_lo[:, i] = lowfi_values(policy.lowfi_values, env.problem, view(s_batch,axes(s_batch)[1:end-1]...,i))
    end
    return q_lo
end