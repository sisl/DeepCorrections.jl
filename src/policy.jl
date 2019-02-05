struct DeepCorrectionPolicy{M<:Union{MDP,POMDP},P<:AbstractNNPolicy, A} <: AbstractNNPolicy
    problem::M
    correction_network::P
    lowfi_values::Any
    correction::Any
    correction_weight::Float64
    action_map::Vector{A}
end

function _actionvalues(policy::DeepCorrectionPolicy, o::AbstractArray{T,N}) where {T<:Real,N}
    q_low = lowfi_values(policy.lowfi_values, policy.correction_network.problem, o)
    q_corr = actionvalues(policy.correction_network, o)
    q_val = correction(policy.correction, policy.correction_network.problem, q_low, q_corr, policy.correction_weight)
    return q_val
end

function POMDPPolicies.actionvalues(policy::DeepCorrectionPolicy{M}, s) where M<:MDP
    return _actionvalues(policy, convert_s(Array{Float64}, s, policy.problem))
end

function POMDPPolicies.actionvalues(policy::DeepCorrectionPolicy{M}, o) where M<:POMDP
    return _actionvalues(policy, convert_o(Array{Float64}, o, policy.problem))
end

function POMDPs.action(policy::DeepCorrectionPolicy, o)
    q_val = actionvalues(policy, o)
    ai = argmax(q_val)
    return policy.action_map[ai]
end

function POMDPs.value(policy::DeepCorrectionPolicy, o)
    return maximum(actionvalues(policy, o))
end

function DeepQLearning.resetstate!(policy::DeepCorrectionPolicy)
    DeepQLearning.resetstate!(policy.correction_network)
end

function DeepQLearning.getnetwork(policy::DeepCorrectionPolicy)
    return getnetwork(policy.correction_network)
end
