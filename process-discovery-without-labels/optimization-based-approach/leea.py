## Limited evaluation evolutionary algorithm ==> Specifically designed for programs with exogenous stochasticity
## This algorithm is based on the concept of fitness inheritance from parent to offspring
## The algorithm handles MIPs (mixed integer programs) comprising Continuous and "Binary" variables
## https://www.cs.ucf.edu/eplex/papers/morse_gecco16.pdf

## Code authored by Dr Abhishek Gupta, Scientist, SIMTech, ASTAR ##

import scipy as sp
import concurrent.futures
from random import sample
import copy
from evolutionary_operators import cxUniform, mutateGaussian, binary_tournament_selection
# Abhishek's advice:
# pop: ~20
# gen: 100
# dim: number of columns
# a budget is going to be 2,000
# f: output a score
# just use CEA_MIP
# a trick to convert 0-1 values to indices by random keys (https://bit.csc.lsu.edu/~jianhua/random-key.pdf)

def LEEA_MIP(f, dim, pop, gen, binary_vars = None, model_info=None, uncertainty_set=None, verbose=False, mutation_rate = None, mutation_stdev = None,
         mutation_damping = None, fitness_decay = None, mini_batch_size = None, parallel_comput = False):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations
    By default, this algorithm assumes objective function maximization

    ## Description of some input arguments ##
    f: function name/handle passed for solution evaluation
    model_info: contains any additional recurring optimization model information (Python Dictionary type)
    uncertainty_set: observed uncertainty set for sample averaged approximation (multi-dimensional array type)
    dim: search space dimensionality
    pop: population size of EA
    gen: number of generations of EA
    binary_vars: List of variable indexes that are Binary-coded
    """
    ############################ List of tunable hyper-parameters in the LEEA ###################################
    if mutation_rate is None:
        mutation_rate = 0.04 # frequency at which parents are mutated to generate offspring
    if mutation_stdev is None:
        mutation_stdev = 0.03 # mutation perturbation
    if mutation_damping is None:
        mutation_damping = 1 # allows mutation strength to dampen with increasing generations, in order to encourage convergence
    if fitness_decay is None:
        fitness_decay = 0.2 # decay rate of inherited fitness
    if mini_batch_size is None:
        mini_batch_size = 2 # subsample size for function evaluations, given an otherwise large uncertainty dataset
    #############################################################################################################

    if binary_vars is not None:
        num_binary_vars = len(binary_vars)
        prob_bitflip = 1/num_binary_vars
        if num_binary_vars > dim or max(binary_vars) >= dim or len(set(binary_vars)) != num_binary_vars:
            raise Exception('Binary variables must all be unique and should not exceed dimensionality of the problem.')
    else:
        num_binary_vars = None
        prob_bitflip = None

    if pop % 2 == 1:
        pop += 1

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch.')

    P = [sp.rand(dim) for _ in range(pop)]
    if binary_vars is not None:
        for i in range(pop):
            subvars = P[i][binary_vars]
            subvars[subvars < 0.5], subvars[subvars>=0.5] = 0, 1
            P[i][binary_vars] = copy.deepcopy(subvars)

    # bestSolutions = []
    population_trajectory = []
    fitnesses_inherited = [0 for _ in range(pop)]
    halfpop = int(pop/2)

    for G in range(gen):
        if (uncertainty_set is not None):
            mini_batch = uncertainty_set[sp.array(sample(batch_indx, mini_batch_size))]
        else:
            mini_batch = None

        if parallel_comput:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                processes = []
                for i in range(pop):
                    proc = executor.submit(f, P[i], model_info, mini_batch)
                    processes.append(proc)
                fitnesses_current = [proc.result() for proc in processes]

        else:
            fitnesses_current = [f(x, model_info, mini_batch) for x in P]

        fitnesses = [fitnesses_current[i]+fitnesses_inherited[i] for i in range(pop)]

        parent_solutions, parent_fitnesses = binary_tournament_selection(pop, P, fitnesses)

        if verbose: print("Step", G,"; average population fitness :", sum(fitnesses)/pop)

        # Offspring creation via uniform crossovers OR mutation ==> as in original LEEA
        P = []
        fitnesses_inherited = []
        for i in range(halfpop):
            if sp.rand(1) > mutation_rate:
                offspring = cxUniform(parent_solutions[i], parent_solutions[i + halfpop],dim,0.5)
                inheritance = (1 - fitness_decay)*(parent_fitnesses[i]+parent_fitnesses[i + halfpop])/2
                fitnesses_inherited.extend([inheritance, inheritance])
            else:
                offspring = mutateGaussian(parent_solutions[i], parent_solutions[i + halfpop], dim, mutation_stdev,
                                           num_binary_vars = num_binary_vars, binary_vars=binary_vars,prob_bitflip=prob_bitflip)
                fitnesses_inherited.extend([(1 - fitness_decay)*parent_fitnesses[i], (1 - fitness_decay)*parent_fitnesses[i + halfpop]])

            P.extend([offspring[0], offspring[1]])

        mutation_stdev = mutation_damping * mutation_stdev

        avgP = sp.array(P).mean(0)
        if binary_vars is not None:
            subvars = avgP[binary_vars]
            subvars[subvars < 0.5], subvars[subvars >= 0.5] = 0, 1
            avgP[binary_vars] = copy.deepcopy(subvars)
        population_trajectory.append(avgP)

    return population_trajectory


def simLEEA_MIP(f, dim, pop, gen, binary_vars=None, model_info=None, uncertainty_set=None, verbose=False,
            mutation_stdev=None, mutation_damping=None, mini_batch_size=None, parallel_comput=False):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations
    By default, this algorithm assumes objective function maximization
    Simplified LEEA (without fitness inheritance mechanisms) seems to work better than LEEA for MIPs of larger dimensionality
    """
    ############################ List of tunable hyper-parameters in the LEEA ###################################
    if mutation_stdev is None:
        mutation_stdev = 0.03  # mutation perturbation
    if mutation_damping is None:
        mutation_damping = 1  # allows mutation strength to dampen with increasing generations, in order to encourage convergence
    if mini_batch_size is None:
        mini_batch_size = 2  # subsample size for function evaluations, given an otherwise large uncertainty dataset
    #############################################################################################################

    if binary_vars is not None:
        num_binary_vars = len(binary_vars)
        prob_bitflip = 1 / num_binary_vars
        if num_binary_vars > dim or max(binary_vars) >= dim or len(set(binary_vars)) != num_binary_vars:
            raise Exception('Binary variables must all be unique and should not exceed dimensionality of the problem.')
    else:
        num_binary_vars = None
        prob_bitflip = None

    if pop % 2 == 1:
        pop += 1

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch.')

    P = [sp.rand(dim) for _ in range(pop)]
    if binary_vars is not None:
        for i in range(pop):
            subvars = P[i][binary_vars]
            subvars[subvars < 0.5], subvars[subvars >= 0.5] = 0, 1
            P[i][binary_vars] = copy.deepcopy(subvars)

    # bestSolutions = []
    population_trajectory = []
    halfpop = int(pop / 2)

    for G in range(gen):
        if (uncertainty_set is not None):
            mini_batch = uncertainty_set[sp.array(sample(batch_indx, mini_batch_size))]
        else:
            mini_batch = None

        if parallel_comput:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                processes = []
                for i in range(pop):
                    proc = executor.submit(f, P[i], model_info, mini_batch)
                    processes.append(proc)
                fitnesses = [proc.result() for proc in processes]

        else:
            fitnesses = [f(x, model_info, mini_batch) for x in P]

        parent_solutions, parent_fitnesses = binary_tournament_selection(pop, P, fitnesses)

        if verbose: print("Step", G, "; average population fitness :", sum(fitnesses) / pop)

        # Offspring creation via uniform crossovers OR mutation ==> as in original LEEA
        P = []
        for i in range(halfpop):
            offspring = cxUniform(parent_solutions[i], parent_solutions[i + halfpop], dim, 0.5)
            offspring = mutateGaussian(offspring[0], offspring[1], dim, mutation_stdev,
                                       num_binary_vars=num_binary_vars, binary_vars=binary_vars,
                                       prob_bitflip=prob_bitflip)

            P.extend([offspring[0], offspring[1]])

        mutation_stdev = mutation_damping * mutation_stdev

        avgP = sp.array(P).mean(0)
        if binary_vars is not None:
            subvars = avgP[binary_vars]
            subvars[subvars < 0.5], subvars[subvars >= 0.5] = 0, 1
            avgP[binary_vars] = copy.deepcopy(subvars)
        population_trajectory.append(avgP)

    return population_trajectory


def CEA_MIP(f, dim, pop, gen, binary_vars=None, model_info=None, verbose=False,
             mutation_stdev=None, mutation_damping=None, parallel_comput=False):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations
    By default, this algorithm assumes objective function maximization
    """
    ############################ List of tunable hyper-parameters in the Canonical EA ###################################
    if mutation_stdev is None:
        mutation_stdev = 0.03  # mutation perturbation
    if mutation_damping is None:
        mutation_damping = 1  # allows mutation strength to dampen with increasing generations, in order to encourage convergence
    #############################################################################################################

    vars = [i for i in range(dim)]
    if binary_vars is not None:
        num_binary_vars = len(binary_vars)
        prob_bitflip = 1 / num_binary_vars
        if num_binary_vars > dim or max(binary_vars) >= dim or len(set(binary_vars)) != num_binary_vars:
            raise Exception('Binary variables must all be unique and should not exceed dimensionality of the problem.')
        continuous_vars = list(set(vars)-set(binary_vars))
    else:
        num_binary_vars = None
        prob_bitflip = None
        continuous_vars = vars

    if pop % 2 == 1:
        pop += 1

    P = [sp.rand(dim) for _ in range(pop)]
    if binary_vars is not None:
        for i in range(pop):
            subvars = P[i][binary_vars]
            subvars[subvars < 0.5], subvars[subvars >= 0.5] = 0, 1
            P[i][binary_vars] = copy.deepcopy(subvars)

    bestFitness = None
    halfpop = int(pop / 2)

    for G in range(gen):

        if parallel_comput:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                processes = []
                for i in range(pop):
                    proc = executor.submit(f, P[i], model_info)
                    processes.append(proc)
                fitnesses = [proc.result() for proc in processes]

        else:
            fitnesses = [f(x, model_info) for x in P]

        maxFitness = max(fitnesses)
        if bestFitness is None:
            bestFitness = maxFitness
            bestSolution = copy.deepcopy(P[sp.argmax(fitnesses)])
            flag = True
        elif maxFitness > bestFitness:
            bestFitness = maxFitness
            bestSolution = copy.deepcopy(P[sp.argmax(fitnesses)])
            flag = True
        else:
            flag = False

        parent_solutions, parent_fitnesses = binary_tournament_selection(pop, P, fitnesses)
        # Incorporate some elitism
        if flag is False:
            parent_solutions[0] = bestSolution
            parent_fitnesses[0] = bestFitness

        if verbose: print("Step", G, "; best fitness found:", bestFitness)

        if G == gen-1:
            break

        # Offspring creation via uniform crossovers OR mutation
        P = []
        for i in range(halfpop):
            offspring = cxUniform(parent_solutions[i], parent_solutions[i + halfpop], dim, 0.5)
            offspring = mutateGaussian(offspring[0], offspring[1], dim, mutation_stdev,
                                    num_binary_vars=num_binary_vars, binary_vars=binary_vars,
                                    prob_bitflip=prob_bitflip)

            P.extend([offspring[0], offspring[1]])

        mutation_stdev = mutation_damping * mutation_stdev

    return bestSolution, bestFitness


################# Testing code #########################################
if __name__ == "__main__":

    def onemax(x, model_info):
        fitness = sum(x)
        return fitness

    CEA_MIP(onemax, 6, 100, 200, binary_vars=[i for i in range(3)], verbose=True,parallel_comput=True,mutation_stdev=0.05)
