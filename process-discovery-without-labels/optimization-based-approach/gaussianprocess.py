import math
import torch
import gpytorch
import mtgp
import stgp
import scipy

def build_stgp(train_x, train_y):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = stgp.ExactGPModel(train_x, train_y, likelihood)

    return likelihood, model


def train_stgp(train_x, train_y, likelihood, model):

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return likelihood, model


def build_mtgp(train_xS, train_xT, train_yS, train_yT):

    # S ==> represents source
    # T ==> represents target

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_i_taskS = torch.zeros(train_xS.size(0),dtype=torch.long)
    train_i_taskT = torch.ones(train_xT.size(0),dtype=torch.long)
    full_train_x = torch.cat([train_xS, train_xT],0)
    full_train_i = torch.cat([train_i_taskS, train_i_taskT],0)
    full_train_y = torch.cat([train_yS, train_yT],0)

    model = mtgp.MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)

    return full_train_x, full_train_i, full_train_y, likelihood, model


def train_mtgp(full_train_x, full_train_i, full_train_y, likelihood, model, verbose = False):

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(50):
        optimizer.zero_grad()
        output = model(full_train_x, full_train_i)
        loss = -mll(output, full_train_y)
        loss.backward()
        if verbose: print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
        optimizer.step()

    return likelihood, model


################# Testing code #########################################
if __name__ == "__main__":
    instSrc = 100
    instTrg = 2

    train_x1 = scipy.rand(instSrc, 2)
    train_x2 = scipy.rand(instTrg, 2)
    train_xS = torch.from_numpy(train_x1)
    train_xT = torch.from_numpy(train_x2)

    train_yS = 1000*(train_xS[:, 0].mul(train_xS[:, 0]) + train_xS[:, 1].mul(train_xS[:, 1]))
    xT = train_xT - 0.1
    train_yT = 100*(xT[:, 0].mul(xT[:, 0]) + xT[:, 1].mul(xT[:, 1]))

    # Build and train transfer learning GP model
    full_train_x, full_train_i, full_train_y, likelihood_TGP, model_TGP = build_mtgp(train_xS, train_xT, train_yS, train_yT)
    likelihood_TGP, model_TGP = train_mtgp(full_train_x, full_train_i, full_train_y, likelihood_TGP, model_TGP)

    # Build and train traditional GP model
    likelihood_GP, model_GP = build_stgp(train_xT, train_yT)
    likelihood_GP, model_GP = train_stgp(train_xT, train_yT, likelihood_GP, model_GP)

    model_TGP.eval()
    likelihood_TGP.eval()
    model_GP.eval()
    likelihood_GP.eval()

    test_x = torch.from_numpy(scipy.array([[0.5, 0.5]]))
    test_i_task = torch.ones(test_x.size(0), dtype=torch.long)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction_TGP = likelihood_TGP(model_TGP(test_x, test_i_task))
        print('TGP mean = ', prediction_TGP.mean, ' and standard deviation = ', prediction_TGP.stddev)
        model_GP = model_GP.double()
        prediction_GP = likelihood_GP(model_GP(test_x.double()))
        print('GP mean = ', prediction_GP.mean, ' and standard deviation = ', prediction_GP.stddev)