import arraymancer
import sequtils, strutils
import random

# PSO object
type PSO* = object
    n_particles, dimensions, n_iterations: int
    w, c1, c2: float
    bounds: seq[float]
    particles, velocity: Tensor[float]
    pbest: Tensor[float]
    gbest: Tensor[float]
    gbest_position: int
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
    result.velocity = zeros[float]([n_particles, dimensions])
    result.particles = randomTensor[float](n_particles, dimensions, 1'f)
    result.pbest = result.particles
    result.pbest_value = newSeq[float](n_particles)
    for i in 0..<result.particles.shape[0]:
        result.pbest_value[i] = Inf
    result.gbest_value = Inf

# trying dummy fitness function
proc fitness*(particles: Tensor[float], dimension: int): float =
    result = 1.0
    for dim in 0..<dimension: 
        result += particles[dim] * particles[dim]
    return result
    # return  (particles[0] * particles[0]) + (particles[1] * particles[1]) + 1.0

# set pbest
proc set_pbest*(p: var PSO) =
    var fitness_candidate: float
    var particles_temp: Tensor[float]
    for i in 0..<p.particles.shape[0]:
        particles_temp = p.particles[i,_].reshape(p.dimensions)
        # fitness_candidate = fitness(p.particles[i,_])
        fitness_candidate = fitness(particles_temp, p.dimensions)
        if p.pbest_value[i] > fitness_candidate:
            p.pbest_value[i] = fitness_candidate
            p.pbest[i,_] = p.particles[i,_]

# set gbest
proc set_gbest*(p: var PSO) =
    var best_fitness_candidate: float
    var particles_temp: Tensor[float]
    for i in 0..<p.particles.shape[0]:
        particles_temp = p.particles[i,_].reshape(p.dimensions)
        # best_fitness_candidate = fitness(p.particles[i,_])
        best_fitness_candidate = fitness(particles_temp, p.dimensions)
        if p.gbest_value > best_fitness_candidate:
            p.gbest_value = best_fitness_candidate
            p.gbest_position = i
            p.gbest = p.particles[i,_].reshape(p.dimensions)

# move particle
proc update*(p: var PSO) =
    for i in 0..<p.particles.shape[0]:
        for j in 0..<p.particles.shape[1]:
            p.particles[i,j] = p.particles[i,j] + p.velocity[i,j]

# move all particles
proc update_particles*(p: var PSO) =
    var new_velocity: float
    var r: float
    for i in 0..<p.particles.shape[0]:
        for j in 0..<p.particles.shape[1]:
            r = rand(max=1.0)
            new_velocity = (p.w*p.velocity[i,j]) + ((p.c1*r) * (p.pbest[i,j] - p.particles[i,j])) + ((p.c2*r) * (p.gbest[j] - p.particles[i,j]))
            p.velocity[i,j] = new_velocity
            p.update()

# check bounds to avoid diverge
proc check_bounds*(p: var PSO) = 
    for i in 0..<p.particles.shape[0]:
        for j in 0..<p.particles.shape[1]:
            if p.particles[i,j] < p.bounds[0]:
                p.particles[i,j] = p.bounds[0]
            if p.particles[i,j] > p.bounds[1]:
                p.particles[i,j] = p.bounds[1]
            

# user input
echo "How many particles? "
var n_particles = readLine(stdin).parseInt()

echo "How many dimensions? "
var dimension = readLine(stdin).parseInt()

echo "How many iterations? "
var n_iterations = readLine(stdin).parseInt()

# testing
let
    # n_particles = 100
    # dimension = 15
    # n_iterations = 200
    w = 0.8
    c1 = 1.496
    c2 = 1.496
    bounds = @[0.0, 1.0]

var pso = optimizerPSO(n_particles, dimension, n_iterations, w, c1, c2, bounds)

# echo "pso: ", pso

var iterations = 0
while iterations < n_iterations:
    echo "iter: ", iterations
    echo "gbest_value: ", pso.gbest_value
    
    pso.set_pbest()
    pso.set_gbest()
    pso.update_particles()
    pso.check_bounds()

    iterations += 1

echo "gbest: ", pso.gbest
# echo "pbest: ", pso.pbest