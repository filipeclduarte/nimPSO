# trabalhando com arraymancer
import arraymancer
import sequtils
import random

# PSO object
type PSO* = object
    n_particles, dimensions, n_iterations: int
    w, c1, c2: float
    bounds: seq[float]
    particles, velocity, pbest, gbest: Tensor[float32]
    pbest_position, gbest_position: seq[int]
    pbest_value: seq[float]
    gbest_value: float

# Initializer
proc optimizerPSO*(n_particles, dimensions, n_iterations: int, 
                      w, c1, c2: float, bounds: seq[float]): PSO =
    result.n_particles = n_particles
    result.dimensions = dimensions
    result.n_iterations = n_iterations
    result.w = w
    result.c1 = c1
    result.c2 = c2
    result.bounds = bounds 
    result.velocity = zeros[float32]([n_particles, dimensions])
    result.particles = randomTensor[float32](n_particles, dimensions, 1'f32)
    result.pbest_value = newSeq[float](n_particles)
    for i in 1..result.particles.shape[0]:
        result.pbest_value[i] = Inf
    result.gbest_value = Inf

# trying dummy fitness function
proc fitness*(particles: Tensor[float32]): float =
    return  (particles[0] * particles[0]) + (particles[1] * particles[1]) + 1.0

# set pbest
proc set_pbest*(p:PSO) =
    var fitness_candidate: float
    for i in 1..p.particles.shape[0]:
        fitness_candidate = fitness(p.particles[i,_])
        if p.pbest_value[i] > fitness_candidate:
            p.pbest_value[i] = fitness_candidate
            p.pbest_position = i
            p.pbest[i,_] = p.particles[i,_]

# set gbest
proc set_gbest*(p:PSO) =
    var best_fitness_candidate: float
    for i in 1..p.particles.shape[0]:
        best_fitness_candidate = fitness(p.particles[i,_])
        if p.gbest_value > best_fitness_candidate:
            p.gbest_value = best_fitness_candidate
            p.gbest_position = i
            p.gbest = p.particles[i,_]

# move particle
proc update*(p:PSO) =
    for i in 1..p.particles.shape[0]:
        p.particles[i,_] = p.particles[i,_] +. p.velocity[i,_]

# move all particles
proc update_particles*(p:PSO) =
    var new_velocity: Tensor[float32]
    var r: float
    for i in 1..p.particles.shape[0]:
        for j in 1..p.particles.shape[1]:
            r = rand(max=1.0)
            new_velocity = (p.w*p.velocity[i,j]) + ((p.c1*r) * (p.pbest[i,j] - p.particles[i,j])) + ((p.c2*r) * (p.gbest[j] - p.particles[i,j]))
            p.velocity[i,j] = new_velocity
            p.update()

# testing
let
    n_particles = 30
    dimension = 2
    n_iterations = 100
    w = 0.9
    c1 = 1.5
    c2 = 1.5
    bounds = @[-1.0, 1.0]

var pso = optimizerPSO(n_particles, dimension, n_iterations, w, c1, c2, bounds)

echo "pso: ", pso

var iterations = 0
while iterations < n_iterations:
    pso.set_pbest()
    pso.set_gbest()

    pso.update_particles()
    iterations += 1

    echo "iteration: ", iterations
    echo "gbest_value: ", pso.gbest_value