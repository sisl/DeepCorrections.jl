# DeepCorrections

[![Build Status](https://travis-ci.org/sisl/DeepCorrections.jl.svg?branch=master)](https://travis-ci.org/sisl/DeepCorrections.jl)

[![Coverage Status](https://coveralls.io/repos/github/sisl/DeepCorrections.jl/badge.svg?branch=master)](https://coveralls.io/github/sisl/DeepCorrections.jl?branch=master)

[![codecov.io](http://codecov.io/github/sisl/DeepCorrections.jl/coverage.svg?branch=master)](http://codecov.io/github/sisl/DeepCorrections.jl?branch=master)

This package implements the deep correction method [1] for solving reinforcement learning problems. The user should define the problem according to the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface. 

[1] M. Bouton, K. Julian, A. Nakhaei, K. Fujimura, and M. J. Kochenderfer, “Utility decomposition with deep corrections for scalable planning under uncertainty,” in International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2018. 

## Installation 

```julia
using Pkg
Pkg.add("POMDPs")
POMDPs.add_registry() # to get DeepQLearning, and RLInterface
Pkg.add(PackageSpec(url="https://github.com/sisl/DeepCorrections.jl"))
``` 

## Usage 

```julia 
using POMDPs
using DeepCorrections
using Flux # for model definition
using DeepQLearning # for underlying DQN solver
using POMDPModels # for gridworld

mdp = SimpleGridWorld()

function my_low_fidelity_values(problem::SimpleGridWorld, s)
    return ones(n_actions(problem)) # dummy example, should return an action value vector 
end

model = Chain(Dense(2, 32, relu), Dense(32, n_actions(mdp))) # input is 2 dimensional, x,y positions in grid world
dqn_solver = DeepQLearningSolver(qnetwork = model, verbose=true) # see DQN docs for all the parameters
solver = DeepCorrectionSolver(dqn = dqn_solver,
                              lowfi_values = my_low_fidelity_values)

policy = solve(solver, problem)
``` 

## Documentation 

The type `DeepCorrectionSolver` relies on the `DeepQLearningSolver` type defined in [DeepQLearning.jl](https://github.com/JuliaPOMDP/DeepQLearning.jl). The deep correction solver supports all the options available in  for the `DeepQLearningSolver`. 

`solve` returns a `DeepCorrectionPolicy` object. It can be used like any policy in the POMDPs.jl interface. 

**Low fidelity value estimation:**

To provide the low fidelity value function to the solver the user can use the `lowfi_values` in the constructor. It can be a function or an object. If this is a function `f`, `f(mdp, s)` will be called to estimate the value. If this is an object `o`, `lowfi_values(o, mdp, s, depth)` will be called.
The output should be a vector of size `n_actions(mdp)`. The actions are ordered accoring to the function `action_index` implemented by the problem writer.

**Correction method:**

Two default correction methods are available:
- additive correction: `Q_lo(s, a) + delta(s, a)`, where `Q_lo` is the result of `lowfi_values` and `delta` is the correction network.
- multiplicative correction: `Q_lo(s, a)delta(s, a)`

An additional constant weight can be used in the correction method using the option `correction_weight` in the solver. The user can write its own correction method via the `correction_method` option. It can be a function or an object. If this is a function `f`, `f(mdp, q_lo, q_corr, correction_weight)` will be called to estimate the value. If this is an object `o`, `correction(o, mdp, q_lo, q_corr, correction_weight)` will be called.

The underlying implementation relies on tensorflow and static graphs, the correction method implemented must support tensor inputs and be tensorflow friendly. The signature should look like this: 
```julia 
    multiplicative_correction(problem::Union{POMDP, MDP}, q_lo::Q, q_corr::Q, weight::Float64) where Q <:Union{Array{Float64}, Tensor}
```

