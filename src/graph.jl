
const Q_SCOPE = "active_q"
const TARGET_Q_SCOPE = "target_q"

"""
    CorrectionTrainGraph
type to store all the tensor of the DQN graph including both the Q network
and the training operations
"""
mutable struct CorrectionTrainGraph
    sess::Session
    s::Tensor
    a::Tensor
    sp::Tensor
    r::Tensor
    done_mask::Tensor
    importance_weights::Tensor
    q_lo::Tensor
    q_lo_p::Tensor
    q::Tensor
    qp::Tensor
    target_q::Tensor
    loss::Tensor
    td_errors::Tensor
    train_op::Tensor
    grad_norm::Tensor
    update_op::Tensor
    lowfi_values::Any
end


"""
Create placeholders for DQN training: s, a, sp, r, done
The shape is inferred from the environment
"""
function build_placeholders(solver::DeepCorrectionSolver, env::AbstractEnvironment)
    obs_dim = obs_dimensions(env)
    n_outs = n_actions(env)
    s = placeholder(Float32, shape=[-1, obs_dim...])
    a = placeholder(Int32, shape=[-1])
    sp = placeholder(Float32, shape=[-1, obs_dim...])
    r = placeholder(Float32, shape=[-1])
    done_mask = placeholder(Bool, shape=[-1])
    w = placeholder(Float32, shape=[-1])
    q_lo = placeholder(Float32, shape=[-1, n_actions(env)])
    q_lo_p = placeholder(Float32, shape=[-1, n_actions(env)])
    return s, a, sp, r, done_mask, w, q_lo, q_lo_p
end


"""
Build the loss operation
relies on the Bellman equation
"""
function build_loss(solver::DeepCorrectionSolver,
                    env::AbstractEnvironment,
                    q_lo::Tensor,
                    q_lo_p::Tensor,
                    q::Tensor,
                    target_q::Tensor,
                    a::Tensor, r::Tensor,
                    done_mask::Tensor,
                    importance_weights::Tensor)
    loss, td_errors = nothing, nothing
    variable_scope("loss") do
        term = cast(done_mask, Float32)
        A = one_hot(a, n_actions(env))
        q_corr_sa = sum(A.*q, 2)
        q_lo_sa = sum(A.*q_lo, 2)
        q_sa = correction(solver.correction, env.problem, q_lo_sa, q_corr_sa, solver.correction_weight)
        q_future = correction(solver.correction, env.problem, q_lo_p, target_q, solver.correction_weight)
        q_samp = r + (1 - term).*discount(env.problem).*maximum(q_future, 2)
        td_errors = q_sa - q_samp
        errors = huber_loss(td_errors)
        loss = mean(importance_weights.*errors)
    end
    return loss, td_errors
end

"""
Build the loss operation with double_q
relies on the Bellman equation
"""
function build_doubleq_loss(solver::DeepCorrectionSolver, env::AbstractEnvironment, q_lo::Tensor, q_lo_p::Tensor, q::Tensor, target_q::Tensor,qp::Tensor, a::Tensor, r::Tensor, done_mask::Tensor, importance_weights::Tensor)
    loss, td_errors = nothing, nothing
    variable_scope("loss") do
        term = cast(done_mask, Float32)
        A = one_hot(a, n_actions(env))
        q_corr_sa = sum(A.*q, 2)
        q_lo_sa = sum(A.*q_lo, 2)
        q_sa = correction(solver.correction, env.problem, q_lo_sa, q_corr_sa, solver.correction_weight)
        best_a = indmax(qp, 2)
        best_A = one_hot(best_a, n_actions(env))
        q_future = correction(solver.correction, env.problem, q_lo_p, target_q, solver.correction_weight)
        target_q_best = sum(best_A.*q_future, 2)
        q_samp = r + (1 - term).*discount(env.problem).*target_q_best
        td_errors = q_sa - q_samp
        errors = huber_loss(td_errors)
        loss = mean(importance_weights.*errors)
    end
    return loss, td_errors
end

function build_graph(solver::DeepCorrectionSolver, env::AbstractEnvironment, graph=Graph())
    sess = init_session(graph)
    s, a, sp, r, done_mask, importance_weights, q_lo, q_lo_p = build_placeholders(solver, env)
    q = build_q(s, solver.dqn.arch, env, scope=Q_SCOPE, dueling=solver.dqn.dueling)
    qp = build_q(sp, solver.dqn.arch, env, scope=Q_SCOPE, reuse=true, dueling=solver.dqn.dueling)
    target_q = build_q(sp, solver.dqn.arch, env, scope=TARGET_Q_SCOPE, dueling=solver.dqn.dueling)
    if solver.dqn.double_q
        loss, td_errors = build_doubleq_loss(solver, env, q_lo, q_lo_p, q, target_q, qp, a, r, done_mask,  importance_weights)
    else
        loss, td_errors = build_loss(solver, env, q_lo, q_lo_p, q, target_q, a, r, done_mask, importance_weights)
    end
    train_op, grad_norm = build_train_op(loss,
                                         lr=solver.dqn.lr,
                                         grad_clip=solver.dqn.grad_clip,
                                         clip_val=solver.dqn.clip_val)
    update_op = build_update_target_op(Q_SCOPE, TARGET_Q_SCOPE)
    return CorrectionTrainGraph(sess, s, a, sp, r, done_mask, importance_weights, q_lo, q_lo_p, q, qp, target_q, loss, td_errors, train_op, grad_norm, update_op, solver.lowfi_values)
end
