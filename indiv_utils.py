"""
indiv_utils.py 
* def sensr_metric - a function to return the sensitive subspace if metric
* def comp_metrics - a function to compare two different mahalanobis metrics

NB: When a function takes 'S' we assume it is already inversed :) 
"""


import copy
import torch
import numpy as np
from scipy.stats import logistic
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import TruncatedSVD

import tensorflow as tf

from deepbayes.analyzers import gradient_expectation

def mahalanobis_dist(x,y,S):
    return distance.mahalanobis(x,y,S)

# This is a very particular noise model choice. This is where we can propose to be smart
def peturbation_generator_1(X,N,D):
    x1 = np.random.randint(0,N,size=(D))
    n1 = np.random.normal(0, 0.1, size=X.shape)
    m1 = np.random.choice([1, 0], size=np.prod(X.shape), p=[0.1, 0.9])   
    m1 = m1.reshape(X.shape)
    return x1, n1, m1

# This is a very particular noise model choice. This is where we can propose to be smart
def peturbation_generator_2(X,N,D):
    x1 = np.random.randint(0,N,size=(D))
    n1 = np.random.normal(0, 0.01, size=X.shape)
    m1 = np.random.choice([1, 0], size=np.prod(X.shape), p=[0.5, 0.5])   
    m1 = m1.reshape(X.shape)
    return x1, n1, m1

def gen_explr_dataset(X, S, pg, thresh=0.1, D=5000):
    N = len(X)
    x1, n1, m1 = pg(X, N, D)
    pairs = []
    labels = []
    dists = []
    for i in range(D):
        i2 = X[x1[i]] + (n1[x1[i]]*m1[i])
        #pairs.append(np.concatenate((X[x1[i]]*0.0, (n1[x1[i]]*m1[i]) )))
        pairs.append(n1[x1[i]]*m1[i])
        dist = mahalanobis_dist(X[x1[i]], i2, S)
        dists.append(dist)
        if(dist < thresh):
            labels.append(1)
        else:
            labels.append(0)
    print("[INFO]: distance distribution information: [mean] %s, [max] %s, [min] %s"%(np.mean(dists), np.max(dists), np.min(dists)))
    return pairs, np.asarray(labels)

def __grad_likelihood__(X, Y, sigma):
    """Computes the gradient of the likelihood function using sigmoidal link"""

    diag = np.einsum("ij,ij->i", np.matmul(X, sigma), X)
    diag = np.maximum(diag, 1e-10)
    prVec = logistic.cdf(diag)
    sclVec = 2.0 / (np.exp(diag) - 1)
    vec = (Y * prVec) - ((1 - Y) * prVec * sclVec)
    grad = np.matmul(X.T * vec, X) / X.shape[0]
    return grad


from numpy import linalg as la
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
def __projPSD__(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def explr_metric(X, Y, iters, batchsize):
    N = X.shape[0]
    P = X.shape[1]

    sigma_t = np.random.normal(0, 1, P ** 2).reshape(P, P)
    sigma_t += np.eye(P)
    sigma_t = np.matmul(sigma_t, sigma_t.T)
    sigma_t = sigma_t / np.linalg.norm(sigma_t)

    curriter = 1

    while curriter < (iters+1):
        batch_idxs = np.random.choice(N, size=batchsize, replace=False)
        X_batch = X[batch_idxs]
        Y_batch = Y[batch_idxs]

        grad_t = __grad_likelihood__(X_batch, Y_batch, sigma_t)
        #grad_t = np.sign(grad_t)
        t = 1.0 / curriter #(1 + curriter // 100)
        sigma_t = __projPSD__(sigma_t - (t * grad_t))
        curriter += 1

    sigma = torch.FloatTensor(sigma_t).detach()
    return sigma

def sensr_metric(data, protected_idxs, keep_protected_idxs=True):
    dtype = torch.Tensor(data).dtype

    # data = datautils.convert_tensor_to_numpy(data)
    basis_vectors_ = []
    num_attr = data.shape[1]

    # Get input data excluding the protected attributes
    protected_idxs = sorted(protected_idxs)
    free_idxs = [idx for idx in range(num_attr) if idx not in protected_idxs]
    X_train = data[:, free_idxs]
    Y_train = data[:, protected_idxs]

    # Update: extended support for continuous target type
    coefs = []
    for idx in range(len(protected_idxs)):
        y_arr = np.array(Y_train[:, idx])
        ttype = type_of_target(y_arr)
        if ttype == 'continuous':
            coefs.append(Lasso()
                         .fit(X_train, Y_train[:, idx])
                         .coef_.squeeze())
        # binary or multiclass or multilabel-indicator (this would break if the ttype is multiclass-multioutput,
        # continuous-multioutput)
        else:
            coefs.append(LogisticRegression(solver="liblinear", penalty="l2", C=10.0)
                         .fit(X_train, Y_train[:, idx])
                         .coef_.squeeze())
    coefs = np.array(coefs)

    if keep_protected_idxs:
        # To keep protected indices, we add two basis vectors
        # First, with logistic regression coefficients with 0 in
        # protected indices. Second, with one-hot vectors with 1 in
        # protected indices.

        basis_vectors_ = np.empty(shape=(2 * len(protected_idxs), num_attr))

        for i, protected_idx in enumerate(protected_idxs):
            protected_basis_vector = np.zeros(shape=(num_attr))
            protected_basis_vector[protected_idx] = 1.0

            unprotected_basis_vector = np.zeros(shape=(num_attr))
            np.put_along_axis(
                unprotected_basis_vector, np.array(free_idxs), coefs[i], axis=0
            )

            basis_vectors_[2 * i] = unprotected_basis_vector
            basis_vectors_[2 * i + 1] = protected_basis_vector
    else:
        # Protected indices are to be discarded. Therefore, we can
        # simply return back the logistic regression coefficients
        basis_vectors_ = coefs

    basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
    basis_vectors_ = basis_vectors_.detach()

    def get_span_of_sensitive_subspace(sensitive_subspace):
        """
        sensitive_subspace: the redundant sensitive subspace
        return: the span of the sensitive subspace
        """
        tSVD = TruncatedSVD(n_components=sensitive_subspace.shape[0])
        tSVD.fit(sensitive_subspace)
        span = tSVD.components_
        return span

    def complement_projector(span):
        """
        span: the span of the sensitive directions
        return: the orthogonal complement projector of the span
        """
        basis = span.T
        proj = np.linalg.pinv(basis.T @ basis)
        proj = basis @ proj @ basis.T
        proj_compl = np.eye(proj.shape[0]) - proj
        return proj_compl

    span = get_span_of_sensitive_subspace(basis_vectors_.T)
    metric_matrix = complement_projector(span)
    metric_matrix += np.eye(len(metric_matrix), dtype=np.float64) * 1e-3  # Inflating the variance to help numerically
    return metric_matrix



def get_bounds_from_mahalanobis(M: np.ndarray) -> np.ndarray:
    """
    Approximates mahalanobis distance interval with axis aligned orthotope
    :param M: a pd matrix
    :return: interval that is an array of lengths such that [-interval, interval] (closely) includes points that are
             unit distance according to MH distance
    """
    l, U = np.linalg.eigh(M)
    ones = np.ones_like(l)
    A = np.matmul(ones/np.sqrt(l), np.abs(U))
    interval_lens = A
    return interval_lens



import copy
from scipy.spatial import distance
def project_to_ellipse(center, point, eps, S, iters=10):
    """
    Project a point from outside of an ellipse to the border of the ellipse (approx)
    """
    d = distance.mahalanobis(center,point,S)
    if(d <= eps):
        return point # if already within distance dont project
    direction = point - center 
    move = -1
    step = 0.5
    p = copy.deepcopy(point)
    for i in range(iters):
        p += move * step *  direction
        d = distance.mahalanobis(center,p,S)
        if(d <= eps and move == -1):
            move = 1
        elif(d >= eps and move == 1):
            move = -1
        step *= 0.5
        #print(d, step)
    #print(' ')
    return p


def _f_PGD(model, inp, loss_fn, eps, S_fair, direc=-1, step=0.1, num_steps=15, num_models=35, order=1):
    input_shape = np.squeeze(inp).shape
    output = model.predict(inp)
    inp_copy = copy.deepcopy(inp)
    
    if(type(direc) == int):
        direc = np.squeeze(model.predict(inp))
        try:
            direc = np.argmax(direc, axis=1)
        except:
            direc = np.argmax(direc)

    adv = np.asarray(inp)
    #maxi = adv + eps; mini = adv - eps
    adv = adv + ((eps/10) * np.sign(np.random.normal(0.0, 1.0, size=adv.shape)))
    adv = np.clip(adv, model.input_lower, model.input_upper)
    #print("PERFORMING PGD")
    for j in range(num_steps+1):
        if(order == 1):
            grad = gradient_expectation(model, adv, direc, loss_fn, num_models)
        #elif(order == 1):
        #    grad = zeroth_order_gradient(model, adv, direc, loss_fn, num_models)
        #grad = grad/np.max(grad, axis=1) #(grad-np.min(grad))/(np.max(grad)-np.min(grad))
        grad = np.sign(grad)
        grad *= np.sqrt(np.diag(np.linalg.inv(S_fair)))
        # Normalize as below if you want to do l2 optimization
        #norm = np.max(grad, axis=1)
        #grad = np.asarray([grad[i]/norm[i] for i in range(len(norm))])
        grad *= (eps/float(num_steps)) # an empirically good learning rate
        adv = adv + grad
        adv = tf.cast(adv, 'float32')
        adv = project_to_ellipse(inp_copy, adv, eps, S_fair)  #np.clip(adv, mini, maxi)
    adv = np.clip(adv, model.input_lower, model.input_upper)
    return adv 


def fPGD(model, inp, loss_fn, eps, S_fair, direction=-1, step=0.1, num_steps=5, num_models=-1, order=1, restarts=0):
    advs = []
    for i in range(restarts+1):
        adv = _f_PGD(model, inp, loss_fn, eps, S_fair, direc=direction, 
                  step=step, num_steps=num_steps, num_models=num_models, order=order)
        advs.append(adv)
    if(restarts == 0):
        return adv
    else:
        return advs





