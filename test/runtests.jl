using DeepCorrections, POMDPModels, DeepQLearning
using Base.Test

test_env_path = joinpath(Pkg.dir(), "DeepQLearning", "test", "test_env.jl")
include(test_env_path)

rng = MersenneTwister(1)
mdp = TestMDP((5,5), 4, 6)
solver = DeepCorrectionSolver(correction_network=QNetworkArchitecture(fc=[8]), 
                            max_steps=10000, lr=0.005, eval_freq=2000,num_ep_eval=100,
                            save_freq = 2000, log_freq = 500,
                            double_q = true, dueling=true, rng=rng)

env = MDPEnvironment(mdp, rng=solver.dqn.rng)

corr_pol = solve(solver, mdp)


# value table 
using DiscreteValueIteration
mdp = GridWorld()
vi_pol = solve(ValueIterationSolver(), mdp)

function vi_values(mdp::GridWorld, s::Array{Float64})
    s_gw = convert_s(GridWorldState, s, mdp)
    si = state_index(mdp, s_gw)
    return vi_pol.qmat[si, :]
end
solver.lowfi_values = vi_values
solver.correction = multiplicative_correction
solver.dqn.eps_fraction = 0. 
solver.dqn.eps_end = 0.01
solver.dqn.max_steps = 20000

corr_pol = solve(solver, mdp)

s = rand(states(mdp))
o = convert_s(Vector{Float64}, s, mdp)
o_batch = reshape(o, (1, size(o)...))
q_corr = run(corr_pol.sess, corr_pol.q, Dict(corr_pol.s => o_batch))
q_vi = vi_pol.qmat[state_index(mdp, s), :]
println(q_corr)
println(q_vi)
println(q_vi[:] + q_corr[:])