## Generalized Committee of Online Evolution Strategies (GECO-ES) ##
## Stochastic Programming via Mixture Modeling Evolution Strategies (SPMMES) ##

## Code authored by Dr Abhishek Gupta, Scientist, SIMTech, ASTAR ##

import scipy as sp
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
import numpy as np
from random import sample
import concurrent.futures
from spes import SPES_RMSprop, SPES_Nesterov, objective_for_NelderMead


def SPMMES_RMSprop(f, x0, Xs, pop, gen, model_info=None, uncertainty_set=None, verbose=False, learningRate = None, learningRate_mcff = None,
                    moving_avg_param = None, perturbation_magnitude = None, perturbation_damping = None, mini_batch_size = None, parallel_comput=False):
    """
    By default, this algorithm assumes objective function maximization

    ## Description of input arguments ##
    f: function name/handle passed for solution evaluation
    x0: unbiased input point for target search initialization (type: array)
    Xs: List of input solutions transferred from source tasks (provides inductive search biases)
    model_info: contains any additional recurring optimization model information (user defined Object)
    uncertainty_set: observed uncertainty set for sample averaged approximation (multi-dimensional array type)
    pop: population size of ES
    gen: number of generations of ES
    """
    Ns = len(Xs) # Number of transferred source task solutions ==> Total number of mixture components is (Ns + 1)
    if Ns == 0:
        population_trajectory = SPES_RMSprop(f, x0, pop, gen, model_info=model_info, uncertainty_set=uncertainty_set, verbose=verbose, learningRate=learningRate,
                                             moving_avg_param=moving_avg_param, perturbation_magnitude=perturbation_magnitude,
                                             perturbation_damping=perturbation_damping,mini_batch_size=mini_batch_size)
        return population_trajectory
    elif pop < 10*(Ns+1):
        raise Exception('Population size too small. Each task should receive at least 10 individuals at the start.')

    dim = len(x0)

    ##################### Tunable hyper-parameters in SPES-RMSprop ########################
    if learningRate is None:
        learningRate = 50*(0.6 * (3 + sp.log(dim)) / dim / sp.sqrt(dim)) # controls step size in gradient-based parameter update ==> RMSprop allows for larger step sizes
    if learningRate_mcff is None:
        learningRate_mcff = 0.001 # controls learning rate of the  mixture coefficients
    if moving_avg_param is None:
        moving_avg_param = 0.9 # Controls the influence of the squared sum of historic gradients in RMSprop
    if perturbation_magnitude is None:
        perturbation_magnitude = 1 # controls variance of multivariate Gaussian distribution used for solution sampling. Large magnitude supports transfers in importance sampling
    if perturbation_damping is None:
        perturbation_damping = 1 # mutation strength decays with increasing generations, in order to encourage convergence
    if mini_batch_size is None:
        mini_batch_size = 2 # subsample size for function evaluations, given an otherwise large uncertainty dataset
    #######################################################################################

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch.')

    ##################
    # bestFitness = -sp.Inf
    # bestFound = None
    ##################
    center0 = x0.copy()
    centers = Xs.copy()
    centers.append(center0)
    mcff = (1/(Ns + 1)) * sp.ones(Ns+1) # array of mixture coefficients
    moving_avgs = [sp.zeros(dim) for _ in range(Ns+1)]
    population_trajectory = []

    for G in range(gen):
        # generate and evaluate samples
        solutions = [] # local samples rescaled to actual solution space
        fitnesses = [] # solution evaluation results
        avg_fitnesses = [] # average fitness of each mixture component

        if (uncertainty_set is not None):
            uncertainty_subset = uncertainty_set[sp.array(sample(batch_indx, mini_batch_size))]

        for i in range(Ns+1):
            # antithetic sampling ==> expected to reduce variance in numerical gradients
            local_pop = int(0.5*mcff[i]*pop)
            local_samples = [sp.randn(dim) for _ in range(local_pop)]
            local_samples = local_samples + [-s for s in local_samples]
            solutions.append([perturbation_magnitude*s + centers[i] for s in local_samples])
            if (uncertainty_set is not None):
                if parallel_comput:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        processes = []
                        for j in range(2*local_pop):
                            proc = executor.submit(f, solutions[i][j], model_info, uncertainty_subset)
                            processes.append(proc)
                        local_fitnesses = [proc.result() for proc in processes]
                else:
                    local_fitnesses = [f(s, model_info, uncertainty_subset) for s in solutions[i]]

                fitnesses.append(local_fitnesses)
                if len(local_fitnesses) > 0:
                    avg_fitnesses.append(sp.array(local_fitnesses).mean())
                else:
                    avg_fitnesses.append(-sp.inf)
            else:
                if parallel_comput:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        processes = []
                        for j in range(2*local_pop):
                            proc = executor.submit(f, solutions[i][j], model_info, uncertainty_set)
                            processes.append(proc)
                        local_fitnesses = [proc.result() for proc in processes]
                else:
                    local_fitnesses = [f(s, model_info, uncertainty_set) for s in solutions[i]]

                fitnesses.append(local_fitnesses)
                if len(local_fitnesses) > 0:
                    avg_fitnesses.append(sp.array(local_fitnesses).mean())
                else:
                    avg_fitnesses.append(-sp.inf)

        if verbose: print("Step", G, "; average population fitness :", max(avg_fitnesses))
        population_trajectory.append(centers[sp.argmax(avg_fitnesses)].copy())

        # Concatenate solutions & fitnesses
        concat_solutions = [item for sublist in solutions for item in sublist]
        concat_fitnesses =  [item for sublist in fitnesses for item in sublist]

        # Fitness function reshaping
        tpop = len(concat_fitnesses) # temp generational population size
        utilities = sp.zeros(tpop)
        shape = sp.maximum(0, sp.log(tpop / 2 + 1.0) - sp.log(1 + sp.array(range(tpop))))
        indx = np.argsort(-sp.array(concat_fitnesses))
        utilities[indx] = shape

        # Construct matrix of mixture component-wise sampling probability densities
        prob_matrix = sp.zeros([tpop,Ns + 1])
        for i in range(Ns+1):
            prob_matrix[:,i] = mvn.pdf(concat_solutions, mean = centers[i], cov = perturbation_magnitude * sp.eye(dim))

        # Gradients via importance sampling (IS) ==> IS also serves to transfer solution / information across components
        prob_vector = np.matmul(prob_matrix, mcff)
        prob_ratio = prob_matrix.copy()
        gradient_mcff = sp.zeros(Ns + 1)
        for i in range(Ns + 1):
            prob_ratio[:, i] = sp.divide(prob_ratio[:, i], prob_vector)
            gradient_mcff[i] = sp.dot(utilities, prob_ratio[:,i])
            gradient = np.matmul(sp.multiply(utilities, prob_ratio[:, i]), (sp.array(concat_solutions) - centers[i]))/pow(perturbation_magnitude, 2)
            moving_avgs[i] = moving_avg_param * moving_avgs[i] + (1 - moving_avg_param) * sp.multiply(gradient, gradient)
            update = learningRate * sp.divide(gradient, sp.sqrt(moving_avgs[i] + 1e-50))
            centers[i] += update

        # Update mixture coefficients
        mcff += learningRate_mcff * gradient_mcff
        mcff = mcff / sum(mcff) # ensures sum of coefficients == 1

    return population_trajectory


def SPMMES_Nesterov(f, x0, Xs, pop, gen, model_info=None, uncertainty_set=None, verbose=False, learning_rate = None, learningRate_mcff = None,
                    momentum_coeff = None, perturbation_magnitude = None, perturbation_damping = None, mini_batch_size = None,parallel_comput=False,
                    dropout = False, dropout_rate = None, dropout_interval = None, dropout_masks = None, do_line_search = False,
                    line_search_size = None, prob_line_search = None):
    """
    By default, this algorithm assumes objective function maximization.
    We recommend to use this solver as opposed to SPMMES_RMSprop as this provides extra features

    ## Description of input arguments ##
    f: function name/handle passed for solution evaluation
    x0: unbiased input point for target search initialization (type: array)
    Xs: List of input solutions transferred from source tasks (provides inductive search biases)
    model_info: contains any additional recurring optimization model information (user defined Object)
    uncertainty_set: observed uncertainty set for sample averaged approximation (multi-dimensional array type)
    pop: population size of ES
    gen: number of generations of ES
    """
    Ns = len(Xs) # Number of transferred source task solutions ==> Total number of mixture components is (Ns + 1)
    if Ns == 0:
        population_trajectory=SPES_Nesterov(f, x0, pop, gen, model_info=model_info, uncertainty_set=uncertainty_set, verbose=verbose, learning_rate=learning_rate,
                                            momentum_coeff=momentum_coeff, perturbation_magnitude=perturbation_magnitude,
                                            perturbation_damping=perturbation_damping,mini_batch_size=mini_batch_size,
                                            parallel_comput=parallel_comput,
                                            dropout=False, dropout_rate=None, dropout_interval=None, dropout_masks=None,
                                            do_line_search=False,
                                            line_search_size=None, prob_line_search=None)
        return population_trajectory
    elif pop < 10*(Ns+1):
        raise Exception('Population size too small. Each task should receive at least 10 individuals at the start.')

    dim = len(x0)

    ##################### Tunable hyper-parameters in SPES-RMSprop ########################
    if learning_rate is None:
        learningRate = (0.6 * (3 + sp.log(dim)) / dim / sp.sqrt(dim))  # controls step size during gradient-based parameter updates ... as per xNES code at https://schaul.site44.com/nes.html
    else:
        learningRate = learning_rate
    if learningRate_mcff is None:
        learningRate_mcff = 0.001 # controls learning rate of the  mixture coefficients
    if momentum_coeff is None:
        momentum_coeff = 0.9  # Coefficient of the momentum term in gradient updates
    if perturbation_magnitude is None:
        perturbation_magnitude = 1 # controls variance of multivariate Gaussian distribution used for solution sampling. Large magnitude supports transfers in importance sampling
    if perturbation_damping is None:
        perturbation_damping = 1 # mutation strength decays with increasing generations, in order to encourage convergence
    if mini_batch_size is None:
        mini_batch_size = 2 # subsample size for function evaluations, given an otherwise large uncertainty dataset
    if dropout:
        if dropout_interval is None:
            dropout_interval = 1  # number of iterations between new variable dropouts
        if dropout_rate is None:
            dropout_rate = 0.8  # gives the probability that a variable will be dropped
    if do_line_search:
        if line_search_size is None:
            line_search_size = 1
        if prob_line_search is None:
            prob_line_search = 1
    #######################################################################################

    if dropout:
        if dropout_masks is None:
            dropout_masks = sp.rand(int(round(gen/dropout_interval)),x0.shape[0])
            dropout_masks[dropout_masks < dropout_rate] = 0
            dropout_masks[dropout_masks >= dropout_rate] = 1
            dropout_masks = list(dropout_masks)  # a list of binary masks representing variable clusters to be optimized together

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch.')

    ##################
    # bestFitness = -sp.Inf
    # bestFound = None
    ##################
    center0 = x0.copy()
    centers = Xs.copy()
    centers.append(center0)
    mcff = (1/(Ns + 1)) * sp.ones(Ns+1) # array of mixture coefficients
    momentums = [sp.zeros(dim) for _ in range(Ns+1)] # list of momentums corresponding to each mixture component
    population_trajectory = []
    mask = sp.ones(dim)

    for G in range(gen):
        # generate and evaluate samples
        solutions = [] # local samples rescaled to actual solution space
        fitnesses = [] # solution evaluation results
        avg_fitnesses = [] # average fitness of each mixture component
        comp_pop = []  # list of population sizes assigned to each mixture component
        if dropout and G % dropout_interval == 0:
            mask = dropout_masks[int(round(sp.rand(1)[0]*(len(dropout_masks)-1)))]
            if learning_rate is None:
                subdim = mask.sum()
                learningRate = (0.6 * (3 + sp.log(subdim)) / subdim / sp.sqrt(subdim))

        if (uncertainty_set is not None):
            uncertainty_subset = uncertainty_set[sp.array(sample(batch_indx, mini_batch_size))]

        for i in range(Ns+1):
            # antithetic sampling ==> expected to reduce variance in numerical gradients
            local_pop = int(0.5*mcff[i]*pop)
            local_samples = [sp.randn(dim) for _ in range(local_pop)]
            local_samples = local_samples + [-s for s in local_samples]
            local_samples = [np.multiply(s, mask) for s in local_samples]
            comp_pop.append(2*local_pop)
            solutions.append([perturbation_magnitude*s + centers[i] + (momentum_coeff*np.multiply(momentums[i],mask)) for s in local_samples])
            if (uncertainty_set is not None):
                if parallel_comput:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        processes = []
                        for j in range(2*local_pop):
                            proc = executor.submit(f, solutions[i][j], model_info, uncertainty_subset)
                            processes.append(proc)
                        local_fitnesses = [proc.result() for proc in processes]
                else:
                    local_fitnesses = [f(s, model_info, uncertainty_subset) for s in solutions[i]]

                fitnesses.append(local_fitnesses)
                if len(local_fitnesses) > 0:
                    avg_fitnesses.append(sp.array(local_fitnesses).mean())
                else:
                    avg_fitnesses.append(-sp.inf)
            else:
                if parallel_comput:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        processes = []
                        for j in range(2*local_pop):
                            proc = executor.submit(f, solutions[i][j], model_info, uncertainty_set)
                            processes.append(proc)
                        local_fitnesses = [proc.result() for proc in processes]
                else:
                    local_fitnesses = [f(s, model_info, uncertainty_set) for s in solutions[i]]

                fitnesses.append(local_fitnesses)
                if len(local_fitnesses) > 0:
                    avg_fitnesses.append(sp.array(local_fitnesses).mean())
                else:
                    avg_fitnesses.append(-sp.inf)

        if verbose:print("Step", G, "; average population fitness :", max(avg_fitnesses))
        population_trajectory.append(centers[sp.argmax(avg_fitnesses)].copy())

        # Concatenate solutions & fitnesses
        concat_solutions = [item for sublist in solutions for item in sublist]
        concat_fitnesses =  [item for sublist in fitnesses for item in sublist]

        # Fitness function reshaping
        tpop = len(concat_fitnesses) # temp generational population size
        utilities = sp.zeros(tpop)
        shape = sp.maximum(0, sp.log(tpop / 2 + 1.0) - sp.log(1 + sp.array(range(tpop))))
        indx = np.argsort(-sp.array(concat_fitnesses))
        utilities[indx] = shape

        # Construct matrix of mixture component-wise sampling probability densities ==> Used for importance sampling
        prob_matrix = construct_probability_matrix(Ns, tpop, G, mask, comp_pop, perturbation_magnitude, concat_solutions, centers, momentum_coeff, momentums)

        # Approximating gradients with importance sampling (IS) ==> IS serves to transfer solution information across the mixture components
        prob_vector = np.matmul(prob_matrix, mcff)
        prob_ratio = prob_matrix.copy()
        gradient_mcff = sp.zeros(Ns + 1)
        for i in range(Ns + 1):
            prob_ratio[:, i] = sp.divide(prob_ratio[:, i], prob_vector)
            gradient_mcff[i] = sp.dot(utilities, prob_ratio[:,i])
            # gradient = np.matmul(sp.multiply(utilities, prob_ratio[:, i]), (sp.array(concat_solutions) - centers[i] -
            #                                                                 (momentum_coeff*np.multiply(momentums[i],mask))))/pow(perturbation_magnitude, 2)
            gradient = np.matmul(utilities, (sp.array(concat_solutions) - centers[i] - (momentum_coeff * np.multiply(momentums[i],mask)))) / pow(perturbation_magnitude, 2)
            if do_line_search and sp.rand(1) <= prob_line_search and i == sp.argmax(avg_fitnesses):
                if uncertainty_set is not None:
                    alpha = minimize(objective_for_NelderMead, sp.array([0]),
                                     args=(f, centers[i] + (momentum_coeff*np.multiply(momentums[i],mask)), gradient, model_info, uncertainty_subset), method='Nelder-Mead')
                else:
                    alpha = minimize(objective_for_NelderMead, sp.array([0]),
                                     args=(f, centers[i] + (momentum_coeff*np.multiply(momentums[i],mask)), gradient, model_info, uncertainty_set), method='Nelder-Mead')
                update = line_search_size * alpha.x[0] * gradient + (momentum_coeff*np.multiply(momentums[i],mask))
                momentums[i] = sp.zeros(dim)
            else:
                # Nesterov Accelerated Gradient-based parameter updates
                update = learningRate * mcff[i] * gradient + (momentum_coeff*np.multiply(momentums[i],mask))
                momentums[i] = update

            centers[i] += update

        # Update other parameters and mixture coefficients
        mcff += learningRate_mcff * gradient_mcff
        mcff = mcff / sum(mcff) # ensures sum of coefficients == 1
        perturbation_magnitude = perturbation_magnitude * perturbation_damping

    return population_trajectory


def construct_probability_matrix(Ns, tpop, G, mask, comp_pop, perturbation_magnitude, concat_solutions, centers, momentum_coeff, momentums):
    prob_matrix = sp.zeros([tpop, Ns + 1])  ## (0.1*(0.999**G)/(Ns+1))*sp.ones([tpop, Ns + 1])
    counter = 0
    for i in range(Ns + 1):
        prob_matrix[counter:counter + comp_pop[i], i] = sp.ones(comp_pop[i]).transpose()
        counter += comp_pop[i]
       ## # if sum(mask) > 500:  # dimensionality may be too large for numerical stability of importance sampling
       ## #     prob_matrix[counter:counter + comp_pop[i], i] = sp.ones(comp_pop[i]).transpose()
       ## #     counter += comp_pop[i]
       ## # else:
       ## #     if perturbation_magnitude >= 1:
       ## #         cov_vec = perturbation_magnitude * mask + (1 - mask) * 0.01
       ## #     else:
       ## #         cov_vec = perturbation_magnitude * (mask + (1 - mask) * 0.01)
       ## #     prob_matrix[:, i] = mvn.pdf(concat_solutions,
       ## #                                 mean=centers[i] + (momentum_coeff * np.multiply(momentums[i], mask)),
       ## #                                 cov=np.diag(cov_vec))
    return prob_matrix


################# Testing code #########################################
if __name__ == "__main__":

    from scipy import dot, array, power, ones
    import numpy
    import matplotlib.pyplot as plt

    # Sphere function with decision variable uncertainty
    def sphere(x, model_info, mini_batch):
        fitness, count = 0, 0
        for i in range(mini_batch.shape[0]):
            fitness += numpy.linalg.norm((x - 50) + mini_batch[i]) ** 2
            count += 1
        return -fitness / count

    dims = 1000
    train_set = 3 * numpy.random.randn(2500, dims)
    test_set = 3 * numpy.random.randn(250, dims)

    population_trajectory_GECOES = SPMMES_Nesterov(sphere, 100 * sp.rand(dims),
                                                   [100 * sp.rand(dims), 100 * sp.rand(dims), 100 * sp.rand(dims),
                                                    100 * sp.rand(dims)], 50, 1000, model_info=None,
                                                   uncertainty_set=train_set, verbose=True,
                                                   parallel_comput=True, do_line_search=False, line_search_size=1,
                                                   prob_line_search=0.2)
    test_fitness = []
    for i in range(len(population_trajectory_GECOES)):
        test_fitness.append(sphere(population_trajectory_GECOES[i], None, test_set))

    plt.plot(test_fitness)
    print(population_trajectory_GECOES[len(population_trajectory_GECOES) - 1].mean())

    plt.ylabel('Performance on Validation Set')
    plt.xlabel('Iterations')
    plt.show()
