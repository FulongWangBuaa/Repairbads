import mne
import numpy as np
import warnings
import matplotlib.pyplot as plt

def nt_vecmult(x,v):
    m,n = x.shape
    v_dim = v.ndim
    # w的维度为1(python默认为行向量)
    if v_dim == 1:
        if m != v.shape[0]:
            raise ValueError('x has the shape',x.shape,'and v has the shape',v.shape)
        else:
            # 修改维度为（n,1）
            v = v.reshape(-1,1)
    # w的维度为2
    else:
        vm,vn = v.shape
        # w必须和x具有相同的大小 或者w和x具有相同的行数或列数
        if vm != m and vn != n:
            raise ValueError('x has the shape',x.shape,'and v has the shape',v.shape)
    x_w = x * v
    return x_w

def nt_demean(x,w=None):
    '''
    remove weighted mean for each channel
    input
        x: data with the shape (times,chans)
        w: weight with the shape (times,chans) or (times,1)
    output
        x_demean: data after demean with the shape (times,chans)
    '''
    m,n = x.shape
    if w is None:
        x_mean = np.mean(np.double(x),axis=0, keepdims=True)
        x_demean = x - x_mean
    else:
        x_w = nt_vecmult(x,w)
        x_mean = np.sum(x_w,axis=0, keepdims=True) / (np.sum(w,axis=0, keepdims=True) + np.finfo(float).eps)

        x_demean = x - x_mean
    return x_demean

def nt_cov(x,w=None):
    if w is None:
        c = np.dot(x.conj().T, x)
        tw = x.shape[0]
    else:
        x_w = nt_vecmult(x,w)
        c = np.dot(x_w.conj().T,x_w)
        tw = np.sum(w.flatten())
    return c,tw

def nt_pcarot(cov,nkeep=None,threshold=None):
    S, V = np.linalg.eigh(cov)
    V = V.real
    S = S.real
    idx = np.argsort(S)[::-1] # reverse sort ev order
    eigenvalues = S[idx]
    topcs = V[:, idx]

    if threshold is not None:
        ii = np.where(eigenvalues/eigenvalues[0]>threshold)[0]
        topcs = topcs[:,ii]
        eigenvalues = eigenvalues[ii]
    if nkeep is not None:
        nkeep=min(nkeep,topcs.shape[1])
        topcs = topcs[:,0:nkeep]
        eigenvalues = eigenvalues[0:nkeep]

    return topcs,eigenvalues

def nt_sns0(c,nneighbors=None,skip=0,wc=None):
    n = c.shape[0]

    if nneighbors is None:
        raise ValueError('need to specify nneighbors')
    if wc is None:
        wc = c
    
    nneighbors = min(nneighbors,n-skip-1)

    r = np.zeros_like(c)
    
    d = np.sqrt(1/(np.diag(c) + np.finfo(float).eps))

    c = nt_vecmult(nt_vecmult(c,d),d.T)
    
    for iChan in range(n):
        c1 = c[:,iChan:iChan+1]
        
        idx = np.argsort(c1 ** 2,axis=0)[::-1]
        
        idx = idx[skip + 1 : skip + 1 + nneighbors]
        idx = np.squeeze(idx)

        c2 = wc[idx][:, idx]
        
        topcs,eigenvalues = nt_pcarot(c2)
        topcs = topcs @ np.diag(1.0 / np.sqrt(eigenvalues))
        topcs[np.isinf(topcs) | np.isnan(topcs)] = 0

        # augment rotation matrix to include this channel
        # topcs = np.vstack([np.array([1, 0]), np.hstack([np.zeros((nneighbors, 1)), topcs])])
        topcs = np.vstack((np.hstack((np.ones((1,1)),np.zeros((1,nneighbors)))),np.hstack((np.zeros((nneighbors,1)),topcs))))
        
        # correlation matrix for rotated data
        new_idx = np.insert(idx, 0, iChan)
        c3 = topcs.T @ wc[new_idx][:,new_idx] @ topcs

        # first row defines projection to clean component iChan
        c4 = c3[0, 1:] @ topcs[1:, 1:].T

        # insert new column into denoising matrix
        r[idx, iChan] = c4

    return r

def nt_sns(data,nneighbors=None,skip=0,w=None,exclude=None):
    from mne.io.pick import _picks_to_idx
    '''
    vy=nt_sns(x,nneigbors,skip,w) - sensor noise suppression

    y: denoised matrix

    x: matrix  to denoise (samples*channels)
    nneighbors: number of channels to use in projection
    skip: number of closest neighbors to skip (default: 0)
    w : weights (default: all ones)

    If w=='auto', the weights are calculated automatically.

    Mean of data is NOT removed.
    '''
    if  isinstance(data, np.ndarray):
        return_raw = False
        m,n = data.shape
        T=False
        if n > m:
            T=True
            x = data.T
    elif isinstance(data, mne.io.Raw) or isinstance(data, mne.io.RawArray):
        return_raw = True
        picks = _picks_to_idx(data.info, picks='mag', exclude=())
        x = data.get_data(picks=picks)
        x = x.T
    else:
        raise ValueError('Input data type not supported')
    
    if nneighbors is None:
        raise ValueError('need to specify nneighbors')
    if w is not None and sum(np.ravel(w))==0:
        raise ValueError('weights are all zero!')
    
    if np.isnan(x).any():
        raise ValueError('x contains NANs')
    if len(np.atleast_1d(nneighbors)) > 1 or len(np.atleast_1d(skip)) > 1:
        raise ValueError('nneighbors and skip must be scalars')

    m,n = x.shape

    c,nc = nt_cov(x)

    TOOBIG = 10
    if w == 'auto':
        y = nt_sns(nt_demean(x),nneighbors,skip)
        d=(y-x)**2
        d1 = np.array([np.mean(x ** 2, axis=0), np.mean(y ** 2, axis=0)])
        d = nt_vecmult(d, 1.0 / np.mean(d1, axis=0,keepdims=True))
        w = d<TOOBIG
        w = np.min(w, axis=1)
        w[np.isnan(w)] = 0
    
    if w is not None:
        wc,nwc = nt_cov(nt_demean(x,w),w=w)
        r = nt_sns0(c,nneighbors=nneighbors,skip=skip,wc=wc)
    else:
        mn1 = 0
        w = np.ones((n))
        r = nt_sns0(c,nneighbors=nneighbors,skip=skip,wc=c)
    # print(r)
    y = x @ r

    if return_raw:
        raw_sns = data.copy()
        raw_sns._data[picks,:] = y.T
        return raw_sns
    else:
        if T:
            y = y.T
        return y