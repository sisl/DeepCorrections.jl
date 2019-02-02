using Revise
using Random
using POMDPs
using Flux
using DeepQLearning
using DeepCorrections
test_env_path = joinpath(dirname(pathof(DeepQLearning)), "..", "test", "test_env.jl")
include(test_env_path)


rng = MersenneTwister(1)
mdp = TestMDP((5,5), 4, 6)
model = Chain(x->flattenbatch(x), Dense(100, 8, tanh), Dense(8, 4))
dqn_solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, learning_rate=0.005, 
                                eval_freq=2000,num_ep_eval=100,
                                log_freq = 500,
                                double_q = false, dueling=true, prioritized_replay=false,
                                rng=rng)
solver = DeepCorrectionSolver(dqn = dqn_solver)
@time policy = solve(solver, mdp)