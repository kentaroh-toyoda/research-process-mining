from scipy import dot, exp, log, sqrt, floor, ones, randn, zeros_like, Inf, argmax, eye, outer, array, power
from scipy.linalg import expm
from random import sample

def xNES(f, x0, pop, gen, model_info=None, uncertainty_set=None,verbose=False, mini_batch_size = None):
    """ Exponential NES (xNES), as described in 
    Glasmachers, Schaul, Sun, Wierstra and Schmidhuber (GECCO'10).
    Maximizes a function f under uncertainty
    """
    if mini_batch_size is None:
        mini_batch_size = 2 # subsample size for function evaluations, given an otherwise large uncertainty dataset

    dim = len(x0)  
    I = eye(dim)
    learningRate = 0.6 * (3 + log(dim)) / dim / sqrt(dim)
    #batchSize = 4 + int(floor(3 * log(dim))) # Originally defined population size
    batchSize = pop
    center = x0.copy()
    A = eye(dim)  # sqrt of the covariance matrix

    if (uncertainty_set is not None):
        batch_size = uncertainty_set.shape[0]
        batch_indx = [i for i in range(batch_size)]
        if batch_size < mini_batch_size:
            raise Exception('Uncertainty set too small. Set should not be smaller than mini batch size.')

    population_trajectory = []
    for G in range(0,gen):
        # produce and evaluate samples
        samples = [randn(dim) for _ in range(batchSize)]

        if (uncertainty_set is not None):
            uncertainty_subset = uncertainty_set[array(sample(batch_indx, mini_batch_size))]
            fitnesses = [f(dot(A, s) + center, model_info, uncertainty_subset) for s in samples]
        else:
            fitnesses = [f(dot(A, s) + center, model_info, uncertainty_set) for s in samples]

        if verbose: print("Step", G,"; average population fitness :", sum(fitnesses)/pop)
        utilities = computeUtilities(fitnesses)
        center += dot(A, dot(utilities, samples))
        covGradient = sum([u * (outer(s, s) - I) for (s, u) in zip(samples, utilities)])
        A = dot(A, expm(0.5 * learningRate * covGradient))

        population_trajectory.append(center.copy())

    return population_trajectory
    

def computeUtilities(fitnesses):
    L = len(fitnesses)
    ranks = zeros_like(fitnesses)
    l = sorted(zip(fitnesses, range(L)))
    #l.sort()
    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = array([max(0., x) for x in log(L / 2. + 1.0) - log(L - array(ranks))])
    utilities /= sum(utilities)       # make the utilities sum to 1
    utilities -= 1. / L  # baseline
    return utilities


################# Testing code #########################################
if __name__ == "__main__":
    
    from scipy import dot, array, power
    import numpy
    
    # Rosenbrock function
    def rosen(x, model_info, uncertainty_set):
        return - sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    def sphere(x, model_info, uncertainty_set):
        x = x
        fitness = numpy.linalg.norm((x - 10)) ** 2
        return -fitness
    
    # example run (30-dimensional Rosenbrock)
    trajectory = xNES(sphere, 0.5*ones(20), 50, 1000, model_info=None, uncertainty_set=None, verbose=True)
    print(trajectory[len(trajectory)-1])
