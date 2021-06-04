# A set of Transfer Bayesian optimization algorithms with non-expensive constraint handling ability #
# The implementations assume function maximization by default. For minimization, please multiply objective by -1 #
# The implementations assume that all decision variables are box-constrained in [0,1] #
import concurrent.futures
import scipy as sp
from scipy.stats import norm
import numpy as np
import torch
import gpytorch
import gaussianprocess
import math
import xnes

# Canonical Bayesian Optimization Algorithm #
def cboa(f_obj, dim, samples, init_sample_size = None, f_con = None):
    if init_sample_size is None:
        init_sample_size = 2*dim + 4

    if samples <= init_sample_size:
        raise Exception('Available sample budget is too small.')

    # Generate and evaluate initial solution samples
    x = sp.rand(init_sample_size, dim) # Initial sample size obtained from refernce "Multiproblem Surrogates IEEE TEVC paper"
    F = generate_init_samples(x, f_obj, init_sample_size)

    f_best = -sp.inf
    for i in range(0, samples-init_sample_size):
        y = sp.array(F)
        y = (y - y.mean())/y.std()
        # build and train Gaussian process model on current data
        train_x = torch.from_numpy(x)
        train_y = torch.from_numpy(y)
        likelihood, model = gaussianprocess.build_stgp(train_x, train_y)
        likelihood, model = gaussianprocess.train_stgp(train_x, train_y, likelihood, model)

        model.eval()
        model = model.double()
        likelihood.eval()
        model_info = [model, likelihood]

        # call surrogate function optimizer and evaluate new generated point
        x_new = xnes.xNES(f_surr, dim, 200, model_info = model_info, f_con = f_con)
        f_new = f_obj(x_new)
        x = np.concatenate((x,sp.array([x_new])),0)
        F.append(f_new)

        # Update best solution found
        if f_new > f_best:
            f_best = f_new
            x_best = x_new

    return x_best, f_best, x, sp.array(F)

# Multi-Problem Surrogate-assisted Transfer Bayesian Optimization Algorithm #
# Based on: "Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J., & Sculley, D. (2017, August). Google Vizier:
# A service for black-box optimization. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge
# discovery and data mining (pp. 1487-1495)." #
def mpsboa(f_obj, dim, samples, xS, yS, init_sample_size=None, f_con = None):
    # xS ==> Source input data (Two-dimensional array type)
    # yS ==> Source output data (one-dimensional array type)
    if init_sample_size is None:
        init_sample_size = 2*dim + 4 # Initial sample size obtained from reference "Multiproblem Surrogates IEEE TEVC paper"

    if samples <= init_sample_size:
        raise Exception('Available sample budget is too small.')

    if xS.shape[1] != dim:
        raise Exception('Source and target feature dimensionality do not match.')

    source_sanitycheck(xS, yS)

    # Build and train source Gaussian process model
    yS = (yS - yS.mean()) / yS.std()
    train_xS = torch.from_numpy(xS)
    train_yS = torch.from_numpy(yS)
    likelihood_S, model_S = gaussianprocess.build_stgp(train_xS, train_yS)
    likelihood_S, model_S = gaussianprocess.train_stgp(train_xS, train_yS, likelihood_S, model_S)
    # Turn source model to evaluation mode
    model_S.eval()
    model_S = model_S.double()
    likelihood_S.eval()

    # Generate and evaluate initial target solution samples
    xT = sp.rand(init_sample_size, dim)
    F = generate_init_samples(xT, f_obj, init_sample_size)

    # Get source model predictions at the target data points
    source_predictions = []
    for x in xT:
        prediction_S = likelihood_S(model_S(torch.from_numpy(sp.array([x])).double()))
        source_predictions.append(prediction_S.mean)

    f_best = -sp.inf
    for i in range(0, samples - init_sample_size):
        yT = sp.array(F)
        yT = (yT - yT.mean()) / yT.std()

        # Compute Error Residuals of source model predictions at the target data points
        for i in range(len(F)):
            yT[i] -= source_predictions[i]

        # build and train Gaussian process model on Residuals data
        train_x = torch.from_numpy(xT)
        train_y = torch.from_numpy(yT)
        likelihood_R, model_R = gaussianprocess.build_stgp(train_x, train_y)
        likelihood_R, model_R = gaussianprocess.train_stgp(train_x, train_y, likelihood_R, model_R)

        model_R.eval()
        model_R = model_R.double()
        likelihood_R.eval()
        model_info = [model_S, likelihood_S, model_R, likelihood_R]

        # call surrogate function optimizer and evaluate new generated point
        x_new = xnes.xNES(f_mpsurr, dim, 200, model_info=model_info, f_con=f_con)
        f_new = f_obj(x_new)
        xT = np.concatenate((xT, sp.array([x_new])), 0)
        F.append(f_new)

        # Get source model prediction at new target data point
        prediction_S = likelihood_S(model_S(torch.from_numpy(sp.array([x_new])).double()))
        source_predictions.append(prediction_S.mean)

        # Update best solution found
        if f_new > f_best:
            f_best = f_new
            x_best = x_new

    return x_best, f_best, xT, sp.array(F)


# Generalized Transfer Bayesian Optimization Algorithm #
# Based on: "Min, A. T. W., Gupta, A., & Ong, Y. S. (2020). Generalizing Transfer Bayesian Optimization to Source-Target
# Heterogeneity. IEEE Transactions on Automation Science and Engineering." #
def gtboa(f_obj, dim, samples, xS, yS, init_sample_size=None, f_con = None):
    # xS ==> Source input data (Two-dimensional array type)
    # yS ==> Source output data (one-dimensional array type)
    if init_sample_size is None:
        init_sample_size = 2*dim + 4 # Initial sample size obtained from reference "Multiproblem Surrogates IEEE TEVC paper"

    if samples <= init_sample_size:
        raise Exception('Available sample budget is too small.')

    # Process and prepare source data for multi-task Gaussian process
    xS = random_feature_map(xS,dim)
    source_sanitycheck(xS, yS)
    train_xS, train_yS = source_to_torch(xS, yS)

    # Generate and evaluate initial target solution samples
    xT = sp.rand(init_sample_size, dim)
    F = generate_init_samples(xT, f_obj, init_sample_size)

    f_best = -sp.inf
    for i in range(0, samples - init_sample_size):
        yT = sp.array(F)
        yT = (yT - yT.mean()) / yT.std()

        # build and train Transfer Gaussian process model
        train_xT = torch.from_numpy(xT)
        train_yT = torch.from_numpy(yT)
        full_train_x, full_train_i, full_train_y, likelihood_TGP, model_TGP  = gaussianprocess.build_mtgp(train_xS, train_xT, train_yS, train_yT)
        likelihood_TGP, model_TGP = gaussianprocess.train_mtgp(full_train_x, full_train_i, full_train_y, likelihood_TGP, model_TGP)

        model_TGP.eval()
        likelihood_TGP.eval()
        model_info = [model_TGP, likelihood_TGP]

        # call generalized surrogate function optimizer and evaluate new generated point
        x_new = xnes.xNES(f_gsurr, dim, 200, model_info=model_info, f_con=f_con)
        f_new = f_obj(x_new)
        xT = np.concatenate((xT, sp.array([x_new])), 0)
        F.append(f_new)

        # Update best solution found
        if f_new > f_best:
            f_best = f_new
            x_best = x_new

    return x_best, f_best, xT, sp.array(F)


# Multi-Source Generalized Transfer Bayesian Optimization Algorithm #
# Extends the above based on: Deisenroth, M., & Ng, J. W. (2015, June). Distributed Gaussian Processes.
# In International Conference on Machine Learning (pp. 1481-1490). PMLR."
def msgtboa(f_obj, dim, samples, xS, yS, init_sample_size=None, f_con = None):
    # xS ==> List of source input data (each element of list must be of two-dimensional array type)
    # yS ==> List of source output data (each element of list must be of one-dimensional array type)
    if init_sample_size is None:
        init_sample_size = 2*dim + 4 # Initial sample size obtained from reference "Multiproblem Surrogates IEEE TEVC paper"

    if samples <= init_sample_size:
        raise Exception('Available sample budget is too small.')

    # Process and prepare multiple source datasets for Transfer Gaussian process modelling
    train_xS, train_yS = [], []
    M = len(xS)
    if len(yS) != M:
        raise Exception("Number of source datasets mismatch.")
    for j in range(M):
        xS[j] = random_feature_map(xS[j], dim)
        source_sanitycheck(xS[j], yS[j])
        t_xS, t_yS = source_to_torch(xS[j], yS[j])
        train_xS.append(t_xS)
        train_yS.append(t_yS)

    # Generate and evaluate initial target solution samples
    xT = sp.rand(init_sample_size, dim)
    F = generate_init_samples(xT, f_obj, init_sample_size)

    f_best = -sp.inf
    for i in range(0, samples - init_sample_size):
        yT = sp.array(F)
        yT = (yT - yT.mean()) / yT.std()

        # build and train a set of Transfer Gaussian process models
        train_xT = torch.from_numpy(xT)
        train_yT = torch.from_numpy(yT)
        likelihood_TGP, model_TGP = [], []
        for j in range(M):
            full_train_x, full_train_i, full_train_y, likelihood, model = gaussianprocess.build_mtgp(train_xS[j], train_xT, train_yS[j], train_yT)
            likelihood, model = gaussianprocess.train_mtgp(full_train_x, full_train_i, full_train_y, likelihood, model)
            likelihood_TGP.append(likelihood.eval())
            model_TGP.append(model.eval())

        model_info = [model_TGP, likelihood_TGP]

        # call multi-source generalized surrogate function optimizer and evaluate newly generated point
        x_new = xnes.xNES(f_msgsurr, dim, 200, model_info=model_info, f_con=f_con)
        f_new = f_obj(x_new)
        xT = np.concatenate((xT, sp.array([x_new])), 0)
        F.append(f_new)

        # Update best solution found
        if f_new > f_best:
            f_best = f_new
            x_best = x_new

    return x_best, f_best, xT, sp.array(F)

def generate_init_samples(x, f_obj, init_sample_size):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processes = []
        for i in range(init_sample_size):
            proc = executor.submit(f_obj, x[i])
            processes.append(proc)
        F = [proc.result() for proc in processes]
    return F


def random_feature_map(xS, dim):
    source_dim = xS.shape[1]
    if source_dim != dim:
        xS = sp.concatenate((xS, sp.ones([xS.shape[0], 1])), 1)
        projection_matrix = sp.rand(dim, source_dim + 1)
        xS = sp.matmul(projection_matrix, xS.transpose()).transpose()  # Simple linear projection
        # Mapping back to range [0,1]
        for i in range(dim):
            xS[:, i] = (xS[:, i] - xS[:, i].min()) / (xS[:, i].max() - xS[:, i].min())
    return xS


def source_to_torch(xS, yS):
    yS = (yS - yS.mean()) / yS.std()
    train_xS = torch.from_numpy(xS)
    train_yS = torch.from_numpy(yS)
    return train_xS, train_yS


def f_surr(x, model_info):
    model = model_info[0]
    likelihood = model_info[1]
    test_x = torch.from_numpy(sp.array([x]))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x.double()))

    return prediction.mean + 0.3*prediction.stddev


def f_gsurr(x, model_info):
    model = model_info[0]
    likelihood = model_info[1]
    test_x = torch.from_numpy(sp.array([x]))
    test_i_task = torch.ones(1, dtype=torch.long)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_x, test_i_task))

    return prediction.mean + 0.3*prediction.stddev


def f_msgsurr(x, model_info):
    models = model_info[0]
    likelihoods = model_info[1]
    M =len(models)
    test_x = torch.from_numpy(sp.array([x]))
    test_i_task = torch.ones(1, dtype=torch.long)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = []
        for j in range(M):
            predictions.append(likelihoods[j](models[j](test_x, test_i_task)))

    predicted_variance = sp.stats.hmean(sp.array([predictions[j].variance for j in range(M)]))
    predicted_mean = 0
    for j in range(M):
        predicted_mean += (predictions[j].mean/predictions[j].variance)
    predicted_mean = predicted_mean*predicted_variance/M

    return predicted_mean + 0.3*math.pow(predicted_variance, 0.5)


def f_mpsurr(x, model_info):
    model_S, likelihood_S = model_info[0], model_info[1]
    model_R, likelihood_R = model_info[2], model_info[3]
    test_x = torch.from_numpy(sp.array([x]))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction_S = likelihood_S(model_S(test_x.double()))
        prediction_R = likelihood_R(model_R(test_x.double()))

    return (prediction_S.mean + prediction_R.mean) + 0.3*prediction_R.stddev #0.3*(math.sqrt(prediction_S.variance + prediction_R.variance)) #


def source_sanitycheck(xS, yS):
    if len(yS.shape) > 1:
        raise Exception('Source output data should be 1-D array.')

    if yS.shape[0] != xS.shape[0]:
        raise Exception('Source input-output size mismatch.')

    if xS.max() > 1.0 or xS.min() < 0.0:
        raise Exception('All source input features should be bounded within [0,1].')

