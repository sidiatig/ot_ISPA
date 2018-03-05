import numpy as np
import  ot

def sinkhorn(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, **kwargs):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    # print(reg)

    K = np.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

     # return OT matrix
    T = u.reshape((-1, 1)) * K * v.reshape((1, -1))
    ones = np.ones(len(a))
    alpha = -reg * np.log(u) + np.sum(np.log(u)) * reg /len(a) * ones
    if log:
        return T, alpha, log
    else:
        return T, alpha

def entropic_barycenters(bs, M, reg, c , max_iter=1000,
                         tol=1e-6, verbose=False, log=False):
    # without quantile, no need to reconstruct the new M, update barycenter with dual prima
    n = bs.shape[0]
    N = bs.shape[1]
    a = np.ones(n, dtype=np.float64) / n

    bs = [np.asarray(bs[:,i], dtype=np.float64) for i in range(N)]
    cpt = 0
    err = 1
    if log:
        log = {'err': []}

    while(err > tol and cpt < max_iter):
        aprev = a
        alphas = [sinkhorn(a, bs[i], M, reg)[1] for i in range(N)]
        alpha = 1 /N * sum(i for i in alphas)
        gradient = np.exp(-1 * c * alpha)
        a = a * gradient
        a = a/np.sum(a)

        err = np.linalg.norm(a - aprev)
        if log:
            log['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1
        print("{}-th iteration, err:{}".format(cpt, err))
        print("min, max of dual barycenter", a.min(), a.max())

    if log:
        return a, log
    else:
        return a

def distance_matrix(x_size, y_size):
    # M with 2d coordinates
    nx, ny = x_size, y_size
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(y, x)
    coors = np.vstack((xv.flatten(), yv.flatten())).T
    coor = np.empty(coors.shape)
    coor[:, 0] = coors[:, 1]
    coor[:, 1] = coors[:, 0]
    M_image = ot.utils.dist(coor)
    return M_image

def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT), weights.T))

def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return np.exp(np.mean(np.log(alldistribT), axis=1))

def bregman_barycenter(A, M, reg, weights=None, numItermax=1000, stopThr=1e-4, verbose=False, log=False):
    if weights is None:
        weights = np.ones(A.shape[1]) / A.shape[1]
    else:
        assert(len(weights) == A.shape[1])

    if log:
        log = {'err': []}

    # M = M/np.median(M) # suggested by G. Peyre
    K = np.exp(-M / reg)

    K[K<1e-300] = 1e-300

    cpt = 0
    err = 1

    UKv = np.dot(K, np.divide(A.T, np.sum(K, axis=0)).T)
    u = (geometricMean(UKv) / UKv.T).T
    bary = geometricBar(weights, UKv)

    while (err > stopThr and cpt < numItermax):
        baryprev = bary
        cpt = cpt + 1
        UKv = u * np.dot(K, np.divide(A, np.dot(K, u)))
        u = (u.T * geometricBar(weights, UKv)).T / UKv
        bary = geometricBar(weights, UKv)
        err = np.linalg.norm(bary - baryprev)
        # barycenter = geometricBar(weights, UKv)
        # print('barycenter of {}th iteration'.format(cpt), barycenter, '---------')
        if cpt % 10 == 1:
            # err = np.sum(np.std(UKv, axis=1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    print(cpt)
    if log:
        log['niter'] = cpt
        return bary, log
    else:
        return bary

