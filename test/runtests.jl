using Test
using DeepCorrections
using Random
using POMDPs
using POMDPModels
using POMDPSimulators
using Flux
using DeepQLearning
using DiscreteValueIteration
Random.seed!(1)

test_env_path = joinpath(dirname(pathof(DeepQLearning)),"..", "test", "test_env.jl")
include(test_env_path)

function evaluate(mdp, policy, rng, n_ep=100, max_steps=100)
    avg_r = 0.
    sim = RolloutSimulator(rng=rng, max_steps=max_steps)
    for i=1:n_ep
        DeepQLearning.resetstate!(policy)
        avg_r += simulate(sim, mdp, policy)
    end
    return avg_r/=n_ep
end

@testset begin "Dummy Corrections"
    rng = MersenneTwister(1)
    mdp = TestMDP((5,5), 4, 6)
    model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, 4))
    dqn_solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                    eval_freq=2000,num_ep_eval=100,
                                    log_freq = 500,
                                    double_q = false, dueling=true, prioritized_replay=false,
                                    rng=rng)
    solver = DeepCorrectionSolver(dqn = dqn_solver)
    corr_pol = solve(solver, mdp)
    r_basic = evaluate(mdp, corr_pol, rng)
    @test r_basic >= 1.5
end

@testset begin "Perfect Qlo"
    rng = MersenneTwister(1)
    # value table 
    mdp = SimpleGridWorld()
    vi_pol = solve(ValueIterationSolver(), mdp)
    function vi_values(mdp::SimpleGridWorld, s::AbstractArray)
        s_gw = convert_s(GWPos, s, mdp)
        si = stateindex(mdp, s_gw)
        return vi_pol.qmat[si, :]
    end
    model = Chain(x->flattenbatch(x), Dense(2, 32, relu), Dense(32, n_actions(mdp)))
    dqn_solver = DeepQLearningSolver(qnetwork = model, prioritized_replay=true, max_steps=40_000, learning_rate=0.01,log_freq=500,
                                     eps_fraction = 0.1, eps_end = 0.01,
                                     double_q=true, dueling=true, rng=rng)
    solver = DeepCorrectionSolver(dqn = dqn_solver, 
                                  lowfi_values = vi_values, 
                                  correction = additive_correction)

    corr_policy = solve(solver, mdp)
    r_gw =  evaluate(mdp, corr_policy, rng)
    @test r_gw >= 0.
end