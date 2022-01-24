## Stochastic Programming Evolution Strategies (SPES) ##
## With acceleration via Nesterov momentum and RMSprop ##

## Code authored by Dr Abhishek Gupta, Scientist, SIMTech, ASTAR ##

import concurrent.futures
import scipy as sp
import numpy as np
from random import sample
from scipy.optimize import minimize


def SPES_Nesterov(f, x0, pop, gen, model_info=None, uncertainty_set=None,verbose=False,learning_rate = None, momentum_coeff = None,
                  perturbation_magnitude = None, perturbation_damping = None, mini_batch_size = None, parallel_comput = False,
                  dropout = False, dropout_rate = None, dropout_interval = None, dropout_masks = None, do_line_search = False,
                  line_search_size = None, prob_line_search = None):
    """
    By default, this algorithm assumes OBJECTIVE FUNCTION MAXIMIZATION

    ## Description of input arguments ##
    f: function name/handle passed for solution evaluation
    x0: input point for target search initialization (array type)
    model_info: contains any additional recurring optimization model information (python Dictionary type)
    uncertainty_set: observed uncertainty set for sample averaged approximation (multi-dimensional array type)
    pop: population size of ES
    gen: number of generations of ES
    """
    dim = len(x0)

    ##################### Tunable hyper-parameters in SPES-Nesterov ########################
    if learning_rate is None:
        learningRate = (0.6 * (3 + sp.log(dim)) / dim / sp.sqrt(dim)) # controls step size during gradient-based parameter updates ... as per xNES code at https://schaul.site44.com/nes.html
    else:
        learningRate = learning_rate
    if momentum_coeff is None:
        momentum_coeff = 0.9 # Coefficient of the momentum term in gradient updates
    if perturbation_magnitude is None:
        perturbation_magnitude = 1 # controls variance of Gaussian distribution used for solution sampling
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

    ########################################################################################

    if dropout:
        if dropout_masks is None:
            dropout_masks = sp.rand(int(round(gen/dropout_interval)),x0.shape[0])
            dropout_masks[dropout_masks < dropout_rate] = 0
            dropout_masks[dropout_masks >= dropout_rate] = 1
            dropout_masks = list(dropout_masks)  # a list of binary masks representing variable clusters to be optimized together

    if pop % 2 == 1:
        pop += 1    

    shape = sp.maximum(0, sp.log(pop/2+1.0) - sp.log(1 + sp.array(range(pop)))) # monotonically decreasing series of smoothed fitness values for fitness shaping

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch.')

    ##################
    # bestFitness = -sp.Inf
    # bestFound = None
    ##################
    center = x0.copy()
    utilities = sp.zeros(pop)
    momentum = sp.zeros(dim)
    population_trajectory = []
    mask = sp.ones(dim)

    for G in range(gen):
        if dropout and G % dropout_interval == 0:
            mask = dropout_masks[int(round(sp.rand(1)[0]*(len(dropout_masks)-1)))]
            if learning_rate is None:
                subdim = mask.sum()
                learningRate = (0.6 * (3 + sp.log(subdim)) / subdim / sp.sqrt(subdim))

        # antithetic sampling and evaluations of solutions
        samples = [sp.randn(dim) for _ in range(int(pop/2))]
        samples = samples + [-s for s in samples]
        samples = [np.multiply(s, mask) for s in samples]
        if do_line_search and sp.rand(1) <= prob_line_search:
            is_memetic = True  # Lifetime learning through local search is to be performed
            momentum = sp.zeros(dim)
        else:
            is_memetic = False

        if (uncertainty_set is not None):
            uncertainty_subset = uncertainty_set[sp.array(sample(batch_indx, mini_batch_size))]
            if parallel_comput:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    processes = []
                    for i in range(pop):
                        proc = executor.submit(f, samples[i] * perturbation_magnitude + center +
                                               (momentum_coeff*np.multiply(momentum, mask)), model_info, uncertainty_subset)
                        processes.append(proc)
                    fitnesses = [proc.result() for proc in processes]
            else:
                fitnesses = [f(perturbation_magnitude*s + center + (momentum_coeff*np.multiply(momentum, mask))
                               , model_info, uncertainty_subset) for s in samples]
        else:
            if parallel_comput:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    processes = []
                    for i in range(pop):
                        proc = executor.submit(f, samples[i] * perturbation_magnitude + center +
                                               (momentum_coeff*np.multiply(momentum, mask)), model_info, uncertainty_set)
                        processes.append(proc)
                    fitnesses = [proc.result() for proc in processes]
            else:
                fitnesses = [f(perturbation_magnitude * s + center +(momentum_coeff*np.multiply(momentum, mask)),
                               model_info, uncertainty_set)
                    for s in samples]

        avg_fitness = sum(fitnesses)/pop
        ###########################################
        # if avg_fitness > bestFitness:
        #     bestFitness = avg_fitness
        #     bestFound = center + (momentum_coeff*momentum)
        ###########################################
        indx = np.argsort(-sp.array(fitnesses))
        utilities[indx] = shape   
        if verbose: print("Step", G,"; average population fitness :", avg_fitness)

        if do_line_search and is_memetic:
            direction = sp.dot(utilities, samples) / perturbation_magnitude
            if uncertainty_set is not None:
                alpha = minimize(objective_for_NelderMead, sp.array([0]), args = (f, center,direction,model_info,uncertainty_subset), method = 'Nelder-Mead')
            else:
                alpha = minimize(objective_for_NelderMead, sp.array([0]),
                                 args=(f, center, direction, model_info, uncertainty_set), method='Nelder-Mead')
            update = line_search_size*alpha.x[0]*direction
        else:
            # Nesterov Accelerated Gradient-based parameter updates
            update = learningRate * sp.dot(utilities, samples) / perturbation_magnitude + (
                        momentum_coeff * np.multiply(momentum, mask))
            momentum = update

        center += update
        perturbation_magnitude = perturbation_magnitude * perturbation_damping

        population_trajectory.append(center.copy())

    return population_trajectory


def objective_for_NelderMead(x, f, center, direction, model_info, uncertainty_set):
    return -f(x*direction + center, model_info, uncertainty_set)


def SPES_RMSprop(f, x0, pop, gen, model_info=None, uncertainty_set=None,verbose=False, learningRate = None,
                    moving_avg_param = None, perturbation_magnitude = None, perturbation_damping = None, mini_batch_size = None, parallel_comput=False):
    """
    By default, this algorithm assumes OBJECTIVE FUNCTION MAXIMIZATION

    ## Description of input arguments ##
    f: function name/handle passed for solution evaluation
    x0: input point for target search initialization (array type)
    model_info: contains any additional recurring optimization model information (python Dictionary type)
    uncertainty_set: observed uncertainty set for sample averaged approximation (multi-dimensional array type)
    pop: population size of ES
    gen: number of generations of ES
    """
    dim = len(x0)

    ##################### Tunable hyper-parameters in SPES-RMSprop ########################
    if learningRate is None:
        learningRate = 50*(0.6 * (3 + sp.log(dim)) / dim / sp.sqrt(dim)) # controls step size in gradient-based parameter update ==> RMSprop allows for larger step sizes
    if moving_avg_param is None:
        moving_avg_param = 0.9 # Controls the influence of the squared sum of historic gradients in RMSprop
    if perturbation_magnitude is None:
        perturbation_magnitude = 1 # controls variance of Gaussian distribution used for solution sampling
    if perturbation_damping is None:
        perturbation_damping = 1 # mutation strength decays with increasing generations, in order to encourage convergence
    if mini_batch_size is None:
        mini_batch_size = 2 # subsample size for function evaluations, given an otherwise large uncertainty dataset
    ########################################################################################

    if pop % 2 == 1:
        pop += 1

    shape = sp.maximum(0, sp.log(pop/2+1.0) - sp.log(1 + sp.array(range(pop)))) # monotonically decreasing series of smoothed fitness values for fitness shaping

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch.')

    ##################
    # bestFitness = -sp.Inf
    # bestFound = None
    ##################
    center = x0.copy()
    utilities = sp.zeros(pop)
    moving_avg = sp.zeros(dim)
    population_trajectory = []

    for G in range(gen):
        # antithetic sampling and evaluations of solutions
        samples = [sp.randn(dim) for _ in range(int(pop/2))]
        samples = samples + [-s for s in samples]

        if (uncertainty_set is not None):
            uncertainty_subset = uncertainty_set[sp.array(sample(batch_indx, mini_batch_size))]
            if parallel_comput:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    processes = []
                    for i in range(pop):
                        proc = executor.submit(f, samples[i] * perturbation_magnitude + center, model_info, uncertainty_subset)
                        processes.append(proc)
                    fitnesses = [proc.result() for proc in processes]
            else:
                fitnesses = [f(perturbation_magnitude * s + center, model_info, uncertainty_subset) for s in samples]
        else:
            if parallel_comput:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    processes = []
                    for i in range(pop):
                        proc = executor.submit(f, samples[i] * perturbation_magnitude + center, model_info, uncertainty_set)
                        processes.append(proc)
                    fitnesses = [proc.result() for proc in processes]
            else:
                fitnesses = [f(perturbation_magnitude * s + center, model_info, uncertainty_set) for s in samples]

        avg_fitness = sum(fitnesses)/pop
        ###########################################
        # if avg_fitness > bestFitness:
        #     bestFitness = avg_fitness
        #     bestFound = center + (momentum_coeff*momentum)
        ###########################################
        indx = np.argsort(-sp.array(fitnesses))
        utilities[indx] = shape
        if verbose: print("Step", G,"; average population fitness :", avg_fitness)

        # RMSprop-based gradient ascent accelerations
        gradient = sp.dot(utilities, samples)/perturbation_magnitude
        moving_avg = moving_avg_param*moving_avg + (1-moving_avg_param)*sp.multiply(gradient, gradient)
        update = learningRate*sp.divide(gradient, sp.sqrt(moving_avg + 1e-50))
        center += update
        perturbation_magnitude = perturbation_magnitude * perturbation_damping

        population_trajectory.append(center.copy())

    return population_trajectory


################# Testing code #########################################
if __name__ == "__main__":
    
    from scipy import dot, array, power, ones
    from xnes import xNES
    import numpy
    import matplotlib.pyplot as plt

    # Sphere function with decision variable uncertainty
    def sphere(x, model_info, mini_batch):
        x = x-20
        fitness, count = 0, 0
        for i in range(mini_batch.shape[0]):
            fitness += numpy.linalg.norm(x + mini_batch[i])**2
            count +=1
        return -fitness/count

    # def sphere_LEEA(x, model_info, mini_batch):
    #     fitness, count = 0, 0
    #     x = x*100
    #     for i in range(mini_batch.shape[0]):
    #         fitness += numpy.linalg.norm((x - 20) + mini_batch[i])**2
    #         count +=1
    #     return -fitness/count

    dims = 20
    train_set = 3*numpy.random.randn(1000,dims)
    test_set = 3*numpy.random.randn(100,dims)

    for run in range(1):
        performances = []
        init = 100 * sp.ones(dims)
    ##########
        # population_trajectory_LEEA = LEEA_MIP(sphere_LEEA, dims, 100, 200, model_info=None,
        #                                              uncertainty_set=train_set, verbose=False,
        #                                              parallel_comput=False)
        # performances.append(-sphere_LEEA(population_trajectory_LEEA[len(population_trajectory_LEEA)-1], None, test_set))
        # LEEA_fitness = []
        # for i in range(len(population_trajectory_LEEA)):
        #     LEEA_fitness.append(-sphere_LEEA(population_trajectory_LEEA[i], None, test_set))
        #
        # plt.plot(LEEA_fitness, label='LEEA')

    ##########
        # population_trajectory_Nesterov = SPES_Nesterov(sphere, init, 100, 200, model_info=None,
        #                                              uncertainty_set=train_set, verbose=False,
        #                                              parallel_comput=False, do_line_search=False)
        # performances.append(-sphere(population_trajectory_Nesterov[len(population_trajectory_Nesterov) - 1], None, test_set))
        # Nesterov_fitness = []
        # for i in range(len(population_trajectory_Nesterov)):
        #     Nesterov_fitness.append(-sphere(population_trajectory_Nesterov[i], None, test_set))
        #
        # plt.plot(Nesterov_fitness, label='Nesterov-ES')

    ##########
        population_trajectory_Memetic = SPES_Nesterov(sphere, init, 20, 1000, model_info=None,
                                                       uncertainty_set=train_set, verbose=False, momentum_coeff= 0.9,
                                                       parallel_comput=False, do_line_search=False, line_search_size=0.2,
                                                       prob_line_search=1)
        # print(population_trajectory_Memetic[len(population_trajectory_Memetic) - 1])
        performances.append(-sphere(population_trajectory_Memetic[len(population_trajectory_Memetic) - 1], None, test_set))
        Memetic_fitness = []
        for i in range(len(population_trajectory_Memetic)):
            Memetic_fitness.append(-sphere(population_trajectory_Memetic[i], None, test_set))

        plt.plot(Memetic_fitness, label='Memetic-ES')

    ##########
        population_trajectory_xNES = xNES(sphere, init, 20, 1000, model_info=None,
                                                      uncertainty_set=train_set, verbose=False)
        performances.append(
            -sphere(population_trajectory_xNES[len(population_trajectory_xNES) - 1], None, test_set))
        xNES_fitness = []
        for i in range(len(population_trajectory_xNES)):
            xNES_fitness.append(-sphere(population_trajectory_xNES[i], None, test_set))

        plt.plot(xNES_fitness, label='xNES')

    ##########
        # population_trajectory_NesterovCoES = SPMMES_Nesterov(sphere, init, [100 * sp.rand(dims),100 * sp.rand(dims),
        #                                             100 * sp.rand(dims),100 * sp.rand(dims)], 100, 200, model_info=None,
        #                                              uncertainty_set=train_set, verbose=False,
        #                                             parallel_comput=False, do_line_search=False)
        # performances.append(-sphere(population_trajectory_NesterovCoES[len(population_trajectory_NesterovCoES) - 1], None, test_set))
        # Memetic_fitness = []
        # for i in range(len(population_trajectory_Memetic)):
        #     Memetic_fitness.append(-sphere(population_trajectory_Memetic[i], None, test_set))
        #
        # plt.plot(Memetic_fitness, label='Memetic-ES')

    ##########
        # population_trajectory_MemeticCoES = SPMMES_Nesterov(sphere, init,
        #                                                      [100 * sp.rand(dims), 100 * sp.rand(dims),
        #                                                       100 * sp.rand(dims), 100 * sp.rand(dims)], 100, 200,
        #                                                      model_info=None,
        #                                                      uncertainty_set=train_set, verbose=False,
        #                                                      parallel_comput=False, do_line_search=True, line_search_size=0.2,
        #                                                     prob_line_search=1)
        # performances.append(-sphere(population_trajectory_MemeticCoES[len(population_trajectory_MemeticCoES) - 1], None, test_set))
        #
        # store_results[run]+= sp.array(performances)
    plt.ylabel('Performance on Validation Set')
    plt.xlabel('Iterations')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.title('Dimensionality = %i' % dims)
    plt.show()
