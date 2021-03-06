import numpy as np

from numba import njit
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

from tqdm import tqdm


# Data import
def compose_set(img_sets, scale):

    dataset = []
    w = [0.2989, 0.5870, 0.1140]

    for set in img_sets:
        iml  = np.dot(set['im_l'][...,:3], w).astype(np.int32)
        imr  = np.dot(set['im_r'][...,:3], w).astype(np.int32)
        disp = (set['im_d'] / scale).astype(np.int32)
        dataset.append([iml, imr, disp])

    return dataset


def get_maxd(set):
    maxd = 0

    for case in set:

        disp = case[2]
        foo = np.max(disp)
        if maxd < foo : maxd = foo

    return maxd


# Main funcs
@njit
def predict(validation_set, q, g, final):

    #print(len(validation_set))

    left  = validation_set[0]
    right = validation_set[1]
    disp  = validation_set[2]

    #print(final," | disp shape: ",disp.shape)

    n = g.size
    h, w = disp.shape

    prediction = np.zeros((h, w), dtype=np.int64)

    for i in range(h):
        F = np.zeros((w, n))
        R = np.zeros((w, n))

        for j in range(1, w):
            penalty = np.ones(n)*np.inf

            for s in range(n):
                for t in range(n):

                    if 0 <= j-1 - t < w:
                        if not final:
                            penalty[t] = F[j-1,t] + q[abs(left[i,j-1] - right[i,j-1-t])] + g[abs(t-s)]
                        if final:
                            penalty[t] = F[j-1,t] + q[abs(left[i,j-1] - right[i,j-1-t])] + g[abs(t-s)] - np.abs(disp[i,j-1-s] - s)
                
                F[j, s] = np.min(penalty)
                R[j, s] = np.argmin(penalty)

                
        last_best = np.zeros((w), dtype=np.int64)
        last = np.ones(n)*np.inf

        # F
        for k in range(n):
            last[k] = F[-1,k] + q[abs(left[i, w-1] - right[i, w-1-t])]
            #print(f">>>Debug {k} | last: {last[k] }")
        last_best[-1] = np.argmin(last) 

        # R
        for j in range(w-1, 0, -1):
            #print(f">>>Debug {j} | last_best: {R[j, last_best[j]]}")
            last_best[j-1] = R[j, last_best[j]]

        # Res
        for k in range(w):
            prediction[i, k] = last_best[k]

    return prediction


def svm(n_iter, dataset, maxd, alpha=-2):
    # cumulatives
    q_cum = np.zeros(256)
    g_cum = np.zeros(maxd+1)

    for iter in tqdm(range(n_iter)):
        q_main = np.zeros_like(q_cum)
        g_main = np.zeros_like(g_cum)
        
        for case in dataset:

            left  = case[0]
            right = case[1]
            gr_tr = case[2]  # ground truth
            guess = predict(case, q_cum, g_cum, True)
            #print(f">>>Debug | case len: {len(case)} | gr_tr: {gr_tr}")

            score_grtr  = score(left, right, gr_tr, q_cum, g_cum)
            score_guess = score(left, right, guess, q_cum, g_cum)

            diff = score_grtr - score_guess + np.sum(np.abs(guess-gr_tr))
            if diff >= 0:
                #print(f">>>Debug | diff: {diff} | iter: {iter}")

                q_best, g_best       = sgrad(left, right, gr_tr, maxd)
                q_predict, g_predict = sgrad(left, right, guess, maxd)

                g_main += g_best - (g_predict / np.linalg.norm(g_predict))
                q_main += q_best - (q_predict / np.linalg.norm(q_predict))

        # final res  
        q_cum -= 2*alpha*q_cum + q_main
        g_cum -= 2*alpha*g_cum + g_main

    return q_cum, g_cum

@njit
def score(left, right, disp, q, g):
    q_, g_ = 0,0

    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):

            g_ += g[abs(disp[i,j]-disp[i,j+1])]
            if 0 <= j - disp[i,j] < disp.shape[1]:
                q_ += q[abs(left[i,j] - right[i,j - disp[i,j]])]

    g_ -= g[abs(disp[disp.shape[0]-1, disp.shape[1]-2]-disp[disp.shape[0]-1,disp.shape[1]-1])]        
    return q_ + g_

@njit
def sgrad(left, right, disp, maxd):
    height, width = disp.shape[:2]

    # max colr diff
    q_ = np.zeros(256)
    # max disp diff
    g_ = np.zeros(maxd+1)   

    for i in range(height):
        for j in range(width):

            for k in range(q_.size):
                if 0 <= j - disp[i,j] < width:
                    q_[k] += int( abs(left[i,j] - right[ i, j-disp[i,j] ]) == k )

            for k in range(g_.size):
                g_[k] += 1. * int( abs(disp[i,j]-disp[i,j+1]) == k ) 

    return q_, g_
