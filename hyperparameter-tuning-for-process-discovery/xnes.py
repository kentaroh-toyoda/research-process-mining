from scipy import dot, exp, log, sqrt, floor, ones, randn, zeros_like, Inf, argmax, eye, outer, array, power, stats
from scipy.linalg import expm

## In this NES implementation, all decision variables are strictly assumed to be bounded in [0,1]

def xNES(f_surr, dim, gen, model_info=None, f_con = None, verbose=False):
    """ Exponential NES (xNES), as described in Glasmachers, Schaul, Sun, Wierstra and Schmidhuber (GECCO'10).
    Maximizes the surrogate objective function f_surr.
    Returns (best solution found, corresponding fitness).
    """
    x0 = 0.5*ones(dim)
    I = eye(dim)
    learningRate = 0.6 * (3 + log(dim)) / dim / sqrt(dim)
    batchSize = 4 + int(floor(3 * log(dim))) # population size
    center = x0.copy()
    A = 0.01*eye(dim)  # sqrt of the covariance matrix
    numEvals = 0
    bestFound = None
    bestFitness = -Inf
    penalty = 1e9 # arbitrarily large multiplier for penalizing constraint violations

    for G in range(0,gen):
        # produce and evaluate samples
        samples = [randn(dim) for _ in range(batchSize)]
        solutions = [dot(A, s) + center for s in samples]

        if f_con is None:
            fitnesses = [f_surr(s, model_info) - penalty * boxcon(s) for s in solutions]
        else:
            fitnesses = [f_surr(s, model_info) - penalty*f_con(s) - penalty*boxcon(s) for s in solutions]

        if max(fitnesses) > bestFitness:
            bestFitness = max(fitnesses)
            bestFound = dot(A,samples[argmax(fitnesses)]) + center
        numEvals += batchSize 
        if verbose: print("Step", G,"; average population fitness :", sum(fitnesses)/batchSize)
        #print A
        # update center and variances
        utilities = computeUtilities(fitnesses)
        center += dot(A, dot(utilities, samples))
        covGradient = sum([u * (outer(s, s) - I) for (s, u) in zip(samples, utilities)])
        A = dot(A, expm(0.5 * learningRate * covGradient))

    return bestFound
    
    
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


def boxcon(x):
    x_temp = x[x<0]
    violation = -sum(x_temp)
    x_temp = (x-1)[(x-1)>0]
    violation += sum(x_temp)
    return violation


################# Testing code #########################################
if __name__ == "__main__":

    import numpy
    
    # Rosenbrock function
    def rosen(x, model_info):
        return - sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

    def sphere(x, model_info):
        x = x*100
        fitness = numpy.linalg.norm((x)) ** 2
        return -fitness

    def f_con(x):
        x = x*100
        x_temp = (x-10)[(x-10)<0]
        return -sum(x_temp)
    
    # example run
    print(xNES(sphere, 5, 10, model_info=None, f_con = f_con, verbose=False))
