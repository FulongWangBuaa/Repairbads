import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans
import numpy as np
import scipy
from scipy.spatial.distance import pdist
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial

# MNE
import mne
from mne.io.pick import _picks_to_idx
from mne.event import make_fixed_length_events
from mne.surface import _normalize_vectors
from mne._fiff.pick import pick_info,_pick_data_channels
from mne.channels.layout import _find_topomap_coords
from mne._ola import _COLA,_Storer
from mne.utils import logger, verbose

# pyriemann

from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance



def _annotate_bad_segments(raw,picks=None,duration=1,overlap=0.1,return_mask=False):
    '''
    根据协方差矩阵之间的黎曼距离来寻找坏段
    输入:
        raw: mne.io.Raw对象
        picks: 通道选择
        duration: 窗口大小, 单位为秒
        overlap: 窗口重叠大小, 单位为秒
        return_mask: 是否返回坏段掩码

    输出:
        raw_annotation: mne.io.Raw对象, 包含坏段标注
    '''
    picks = _picks_to_idx(raw.info, picks=picks, exclude=())

    data = raw._data[picks,:]
    n_chans,n_samples = data.shape
    n_duration = int(round(float(duration) * raw.info["sfreq"]))
    n_overlap = int(round(float(overlap) * raw.info["sfreq"]))

    s = list(range(0, len(raw.times), n_duration-n_overlap)) + [len(raw.times)]
    starts = []
    stops = []
    for i in s:
        if i+n_duration < n_samples:
            starts.append(i)
            stops.append(i+n_duration)
    starts.append(n_samples-n_duration)
    stops.append(n_samples)

    n_segments = len(starts)
    data_segment = np.zeros((n_segments,n_chans,n_duration))
    for idx,(start, stop) in enumerate(zip(starts, stops)):
        data_segment[idx,:,:] = data[:,start:stop]

    covs = covariances(data_segment,estimator="cov")
    
    w = np.ones((n_segments))
    dis = np.zeros((n_segments))
    mask = np.ones((n_samples))
    segments_idx = []
    while True:
        cov_mean = get_mean_cov(covs,metric="riemann",w=w)
        dis = get_distances(covs, cov_mean,metric='riemann')
        
        dis_pick_idx = np.where(w==1)[0]
        dis_pick = [dis[idx] for idx in dis_pick_idx]

        mean = geometric_mean(dis_pick)
        std = geometric_std(dis_pick)
        th = mean+std

        segments_idx = np.where(dis_pick>th)[0]

        if len(segments_idx)==0:
            cov_mean = get_mean_cov(covs,metric="riemann",w=w)
            dis = get_distances(covs, cov_mean,metric='riemann')

            mean = geometric_mean(dis,w)
            std = geometric_std(dis,w)
            th = mean+std

            segments_idx = np.where(dis>th)[0]

            for i in segments_idx:
                mask[starts[i]:stops[i]] = 0
            break

        else:
            for i in segments_idx:
                w[dis_pick_idx[i]] = 0

    raw_annotation = annotate_raw_by_mask(raw,mask)

    if return_mask:
        return raw_annotation,mask
    else:
        return raw_annotation

def generate_colors(num_colors):
    '''
    生成指定数量的颜色
    输入:
        num_colors: 需要的颜色数量
    输出:
        colors: RGBA颜色
    '''
    import matplotlib.colors as mcolors
    XKCD_Colors = mcolors.XKCD_COLORS
    color_name = list(XKCD_Colors.keys())
    color_name.reverse()
    colors = []
    for i in range(num_colors):
        color_hex = XKCD_Colors[color_name[i]]
        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        a = 255 / 255.0
        color_rgba = (r,g,b,a)
        colors.append(color_rgba)
    return colors


def sensor_segmentation(raw,n_clusters=None,plot=False):
    '''
    根据kmeans来划分不同传感器的子集
    输入:
        raw: mne.io.Raw对象
        n_clusters: kmeans聚类数量, 默认为None, 自动根据传感器数量决定
        plot: 是否绘制传感器子集图
    输出:
        ch_segments: 传感器子集列表
    '''
    import matplotlib.colors as mcolors
    info = raw.info
    picks = _pick_data_channels(info, exclude=())
    n_chans = len(picks)
    if n_clusters is None:
        if n_chans < 10:
            return picks
        else:
            n_clusters = int(n_chans / 10)
    pos = _find_topomap_coords(info, picks=picks, sphere=None,ignore_overlap=True)
    pos = pos[:, :2]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init='auto').fit(pos)
    labels = kmeans.labels_
    ch_segments = []
    for i in np.unique(labels):
        idx = np.where(labels == i)[0]
        ch_segments.append(picks[idx])
    if plot:
        colors = [(1, 1, 1)] * 256 
        cmap = mcolors.ListedColormap(colors)
        colors = generate_colors(n_clusters)
        colors = ['b','r','lime','cyan','yellow','magenta']
        fig,ax = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
        mne.viz.plot_topomap(np.zeros(n_chans),info,axes=ax,cmap=cmap)
        pos_x, pos_y = pos.T
        for color,chs in zip(colors,ch_segments):
            for ch_idx in chs:
                mask_params = dict(marker='o', markerfacecolor=color, markeredgecolor='k',
                linewidth=0, markersize=5)
                ax.plot(pos_x[ch_idx], pos_y[ch_idx], **mask_params)
        plt.show()
    return ch_segments

def annotate_raw_by_mask(raw,mask):
    '''
    根据坏段掩码, 标注坏段
    输入:
        raw: mne.io.Raw对象
        mask: 坏段掩码
    输出:
        raw_annotation: mne.io.Raw对象, 包含坏段标注
    '''
    bads_idx = np.where(mask==0)[0]
    if len(bads_idx)==0:
        return raw
    else:
        bad_segments = find_consecutive_numbers(bads_idx)
        n_bad_segments = len(bad_segments)

        onset = np.zeros((n_bad_segments),dtype=float)
        duration = np.zeros((n_bad_segments),dtype=float)
        description = np.array(['bads']*n_bad_segments)
        for i,bad_segment in enumerate(bad_segments):
            start_time = raw.times[bad_segment[0]]
            stop_time = raw.times[bad_segment[-1]]
            onset[i] = start_time
            duration[i] = stop_time - start_time

        annotations = mne.Annotations(onset, duration, description)
        raw_annotation = raw.set_annotations(annotations)
        return raw_annotation

def find_consecutive_numbers(input_list):
    '''
    找到连续的数字段
    '''
    result = []
    temp = []
    for i in range(len(input_list)):
        if i == 0 or input_list[i] != input_list[i-1] + 1:
            if temp:
                result.append(temp)
                temp = []
        temp.append(input_list[i])
    if temp:
        result.append(temp)
    return result

def get_mean_cov(covs,metric,w=None):
    
    if w is None:
         w = np.ones((covs.shape[0]))
    idx = np.where(w==1)[0]
    cov_mean = mean_covariance(covs[idx,:,:],metric=metric)
    return cov_mean

def get_distances(covs, cov_mean,metric):
    return [distance(mtr, cov_mean,metric) for mtr in covs]  

def geometric_mean(datarr,w=None):
    if w is None:
         w = np.ones((len(datarr)))
    idx = np.where(w==1)[0]
    datarr = [datarr[i] for i in idx]
    return np.exp(sum([np.log(x) for x in datarr])/len(datarr))

def geometric_std(datarr,w=None):
    if w is None:
         w = np.ones((len(datarr)))
    m = geometric_mean(datarr,w)
    idx = np.where(w==1)[0]
    datarr = [datarr[i] for i in idx]
    tmp = [np.log(x/m) for x in datarr]
    tmp = sum([t*t for t in tmp])/len(tmp)
    return np.exp(np.sqrt(tmp))


def annotate_bad_segments(raw,duration=1,overlap=0.1,plot=False):
    '''
    根据kmeans自动划分传感器为多个子集,并使用多个黎曼流行对每个子集进行坏段检测
    输入:
        raw: mne.io.Raw对象
        duration: 窗口大小, 单位为秒
        overlap: 窗口重叠大小, 单位为秒
    输出:
        raw_annotation: mne.io.Raw对象, 包含坏段标注
    '''
    ch_segments = sensor_segmentation(raw,plot=plot)
    n_samples = len(raw.times)
    mask = np.ones((n_samples))
    for chs in ch_segments:
        _,m = _annotate_bad_segments(raw.copy().pick(picks=chs),
                                                duration=duration,
                                                overlap=overlap,
                                                return_mask=True)
        mask = np.minimum(mask,m)

    raw_annotation = annotate_raw_by_mask(raw,mask)

    return raw_annotation

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

def nt_normcol(x,w=None):
    m,n = x.shape
    if w is None:
        N = np.sum(x**2,axis=0,keepdims=True)/m
        NN = N**(-0.5)
        if isinstance(N, np.ndarray):
            idx = np.where(N == 0)[0]
            NN[idx] = 0
        else:
            if N == 0:
                NN = 0
        y = nt_vecmult(x,NN)
    else:
        if w.shape[0] != x.shape[0]:
            raise ValueError('weight matrix should have same ncols as data')
        N = np.sum(x**2 * w,axis=0,keepdims=True) / np.sum(w,axis=0,keepdims=True)
        NN = N **(-0.5)
        if isinstance(N, np.ndarray):
                idx = np.where(N == 0)[0]
                NN[idx] = 0
        else:
            if N == 0:
                NN = 0
        y = nt_vecmult(x,NN)
    return y

def nt_pca(x,nkeep=None,threshold=None,w=None):

    if nkeep is None:
        nkeep = None
    if threshold is None:
        threshold = None
    if w is None:
        w = None

    c = nt_cov(x,w)[0]
    topcs,evs = nt_pcarot(c,nkeep,threshold)

    z = np.dot(x,topcs)
    return z

def nt_cov(x,w=None):
    if w is None:
        c = np.dot(x.conj().T, x)
        tw = x.shape[0]
    else:
        x_w = nt_vecmult(x,w)
        c = np.dot(x_w.conj().T,x_w)
        tw = np.sum(w.flatten())
    return c,tw

def nt_xcov(x,y,w=None):
    if w is not None and x.shape[0] != w.shape[0]:
        raise ValueError('x and w must have same nrows')
    mx,nx = x.shape
    my,ny = y.shape

    c = np.dot(x.conj().T, y)
    if w is None:
        tw = y.shape[0]
    else:
        w = w[0:y.shape[0],:]
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

def nt_dss0(c0,c1,keep1=None,keep2=None):
    if keep2 is None:
        keep2 = 1e-9
    if keep1 is None:
        keep1 = None
    if c1 is None:
        raise ValueError('needs at least two arguments')
    
    if c0.shape != c1.shape:
        raise ValueError("C0 and C1 should have the same size")
    if c0.shape[0] != c0.shape[1]:
        raise ValueError("C0 should be square")
    
    if np.isnan(c0).any():
        raise ValueError('NaN in c0')
    if np.isnan(c1).any():
        raise ValueError('NaN in c1')
    if np.isinf(c0).any():
        raise ValueError('INF in c0')
    if np.isinf(c1).any():
        raise ValueError('INF in c1')
    
    topcs1,evs1=nt_pcarot(c0,keep1,keep2)
    evs1=abs(evs1)

    if keep1 is not None:
        topcs1 = topcs1[:, :keep1]
        evs1 = evs1[:keep1]
    if keep2 is not None:
        idx = np.where(evs1 / max(evs1) > keep2)[0]
        topcs1 = topcs1[:, idx]
        evs1 = evs1[idx]

    # PCA and whitening matrix from the unbiased covariance
    N = np.diag(1 / np.sqrt(evs1))

    c2 = N.T @ topcs1.T @ c1 @ topcs1 @ N

    # matrix to convert PCA-whitened data to DSS
    topcs2,evs2=nt_pcarot(c2,keep1,keep2)

    # DSS matrix (raw data to normalized DSS)
    todss = topcs1 @ N @ topcs2
    N2 = np.diag(todss.T @ c0 @ todss)
    todss = todss @ np.diag(1 / np.sqrt(N2))  # adjust so that components are normalized

    # power per DSS component
    pwr0 = np.sqrt(np.sum((c0.T @ todss) ** 2, axis=0))  # unbiased
    pwr1 = np.sqrt(np.sum((c1.T @ todss) ** 2, axis=0))  # biased

    return todss,pwr0,pwr1

def nt_regcov(cxy,cyy,keep=None,threshold=None):
    # PCA of regressor
    topcs,eigenvalues=nt_pcarot(cyy)
    
    # discard negligible regressor PCs
    if keep is not None:
        keep=max(keep,topcs.shape[1])
        topcs = topcs[:,0:keep]
        eigenvalues = eigenvalues[0:keep]

    if threshold is not None:
        idx = np.where(eigenvalues/max(eigenvalues)>threshold)[0]
        topcs = topcs[:,idx]
        eigenvalues = eigenvalues[idx]
    # cross-covariance between data and regressor PCs
    cxy = cxy.T
    r = np.dot(topcs.T, cxy)
    # projection matrix from regressor PCs
    r = nt_vecmult(r, 1/eigenvalues.conj().T)
    # projection matrix from regressors
    r = np.dot(topcs, r)
    return r

def nt_tsr(x,ref,wx=None,wref=None,keep=None,thresh=None):
    mx,nx = x.shape
    mref,nref = ref.shape
    w = np.zeros((mx,1))
    if wx is None and wref is None:
        w[0:,:] = 1
    wx = w
    wref = w

    # remove weighted means
    # x0 = x
    # x = nt_demean(x,wx)
    # mn1 = x - x0
    # ref = nt_demean(ref,wref)

    # equalize power of ref chans, then equalize power of ref PCs
    ref = nt_normcol(ref,wref)
    ref = nt_pca(ref,threshold=1e-6)
    ref=nt_normcol(ref,wref)

    # covariances and cross covariance with refs
    cref,twcref=nt_cov(ref,wref)
    cxref,twcxref=nt_xcov(x,ref,wx)

    # regression matrix of x on refs
    r = nt_regcov(cxref/twcxref,cref/twcref,keep,thresh)

    z = np.dot(ref,r)
    y = x - z

    # y = nt_demean(y,wx)
    return y


def _repair_bad_segments(raw_repair,nremove=1):
    '''
    坏段修复
    输入：
        raw: mne.io.Raw对象
        nremove: 要移除的成分数量

    输出：
        raw_repair: mne.io.Raw对象, 修复后的数据
    '''
    # logger.info("Repairing bad segments")
    picks = _picks_to_idx(raw_repair.info, picks=None, exclude='bads')

    annotations = raw_repair.annotations
    sfreq = raw_repair.info["sfreq"]

    if len(annotations) != 0:
        data_good = raw_repair.get_data(picks=picks,reject_by_annotation='omit',verbose=False)
        data_good = data_good.T
        c0 = np.dot(data_good.T,data_good)
        
        for annotation in annotations:
            
            start = int(annotation['onset'] * sfreq) - raw_repair.first_samp
            stop = int((annotation['onset'] + annotation['duration']) * sfreq) - raw_repair.first_samp

            data_bad = raw_repair._data[picks,start:stop]

            data_bad = data_bad.T
            c1 = np.dot(data_bad.T,data_bad)

            todss,pwr0,pwr1 = nt_dss0(c0,c1)

            p = pwr1/pwr0

            # artifact = np.dot(data_bad,todss[:,idx])

            artifact = np.dot(data_bad,todss[:,0:nremove])

            data_bad_repair=nt_tsr(data_bad,artifact)
            raw_repair._data[picks,start:stop] = data_bad_repair.T

    raw_repair.set_annotations(mne.Annotations([], [], []))

    return raw_repair


def repair_bad_segments(raw,return_annotation_raw=False,plot=False):
    raw_repair = raw.copy()
    iter = 0
    while True:
        print("Iteration:",iter+1)
        raw_annotation = annotate_bad_segments(raw_repair)
        if iter == 0:
            raw_annotation_first = raw_annotation.copy()
        if plot:
            raw_annotation.plot()
            plt.show()
        
        annotations = raw_annotation.annotations
        if len(annotations) == 0 or iter>3:
            if len(annotations) == 0:
                print("Done")
            else:
                print("Maximum iteration reached")
            raw_repair = raw_annotation.set_annotations(mne.Annotations([], [], []))
            break
        else:
            raw_repair = _repair_bad_segments(raw_annotation)
            iter += 1
    if return_annotation_raw:
        return raw_repair,raw_annotation_first
    else:
        return raw_repair


























def _DDWiener(data):
    # Compute the sample covariance matrix
    C = np.dot(data, data.T)
    # Compute the trace of the covariance matrix
    gamma = np.mean(np.diag(C))

    # Compute the DDWiener estimates in each channel
    y_solved = np.zeros_like(data)
    chanN = data.shape[0]
    for i in range(chanN):
        idiff = np.setdiff1d(np.arange(chanN), i)
        y_solved[i,:] = C[i,idiff] @ np.linalg.inv(C[idiff,:][:,idiff] + gamma * np.eye(chanN-1)) @ data[idiff,:]
    sigmas = np.sqrt(np.diag(np.dot(data-y_solved,(data-y_solved).T)))/np.sqrt(data.shape[1])
    return y_solved, sigmas

def _sound(data, LFM, iter, lambda0=0.01):
    """Perform SOUND on one segment of data."""
    chanN = data.shape[0]
    sigmas = np.ones((chanN,1))

    # y_solved, sigmas = _DDWiener(data)
    # Compute the sample covariance matrix
    C = np.dot(data, data.T)
    # Compute the trace of the covariance matrix
    gamma = np.mean(np.diag(C))
    # Compute the DDWiener estimates in each channel
    y_solved = np.zeros_like(data)
    chanN = data.shape[0]
    for i in range(chanN):
        # idiff = np.setdiff1d(np.arange(chanN), i)
        idiff = np.array([j for j in np.arange(chanN) if j != i])
        # y_solved[i,:] = C[i,idiff] @ np.linalg.inv(C[idiff,:][:,idiff] + gamma * np.eye(chanN-1)) @ data[idiff,:]
        y_solved[i,:] = np.dot(np.dot(C[i,idiff], np.linalg.inv(C[idiff,:][:,idiff] + gamma * np.eye(chanN-1))),data[idiff,:])
    error = data-y_solved
    sigmas = np.sqrt(np.diag(np.dot(error,error.T)))/np.sqrt(data.shape[1])
    
    # Number of time points
    T = data.shape[1]
    # Going through all the channels as many times as requested
    dn = np.empty(shape=(0,))
    if isinstance(iter, int):
        for k in range(iter):
            # print("Iteration %s" % (k+1))s
            sigmas_old = sigmas.copy()
            # Evaluating each channel in a random order
            random_ch = np.random.permutation(chanN)
            for i in random_ch:
                
                chan =np.array([j for j in np.arange(chanN) if j != i])
                # Defining the whitening operator with the latest noise estimates
                W = np.diag(1/sigmas)
                # Computing the whitened version of the lead field
                WL = np.dot(W[chan,:][:,chan],LFM[chan,:])
                WLLW = np.dot(WL,WL.T)

                # Computing the MNE, the Wiener estimate in the studied channel
                # as well as the corresponding noise estimate
                # eq.(4)
                lamb = lambda0*np.trace(WLLW)/((chanN-1))
                # 类似于空间滤波器
                P = np.dot(np.dot(WL.T, np.linalg.inv(WLLW + lamb*np.eye(chanN-1))), W[chan,:][:,chan])
                
                # eq.(3)、eq.(6)
                y_solved = np.dot(np.dot(LFM, P), data[chan,:])
                # eq.(5)
                sigmas[i] = np.sqrt(np.sum((y_solved[i,:]-data[i,:])**2)/T)

            # Following and storing the convergence of the algorithm
            dn = np.append(dn,np.max(np.abs(sigmas_old - sigmas)/sigmas_old))

    elif isinstance(iter,str):
        k=0
        while True:
            if len(dn) > 0 and (dn[-1] < 0.01 or k == 100):
                break
            else:
                k = k + 1
                # print("Iteration %s" % (k))

            sigmas_old = sigmas.copy()
            # Evaluating each channel in a random order
            random_ch = np.random.permutation(chanN)
            for i in random_ch:
                
                # chan = np.setdiff1d(np.arange(chanN), i)
                chan = np.array([j for j in np.arange(chanN) if j != i])
                # Defining the whitening operator with the latest noise estimates
                W = np.diag(1/sigmas)
                # Computing the whitened version of the lead field
                WL = np.dot(W[chan,:][:,chan],LFM[chan,:])
                WLLW = np.dot(WL,WL.T)

                # Computing the MNE, the Wiener estimate in the studied channel
                # as well as the corresponding noise estimate
                # eq.(4)
                # if isinstance(lambda0,float):
                lamb = lambda0*np.trace(WLLW)/((chanN-1))
                # elif isinstance(lambda0,str):
                #     lamb = np.trace(WLLW)/np.trace(np.dot(y_solved,y_solved.T)/T)
                # print(lamb)
                # 类似于空间滤波器
                P = np.dot(np.dot(WL.T, np.linalg.inv(WLLW + lamb*np.eye(chanN-1))), W[chan,:][:,chan])
                
                # eq.(3)、eq.(6)
                y_solved = np.dot(np.dot(LFM, P), data[chan,:])
                # eq.(5)
                sigmas[i] = np.sqrt(np.sum((y_solved[i,:]-data[i,:])**2)/T)

            # Following and storing the convergence of the algorithm
            dn = np.append(dn,np.max(np.abs(sigmas_old - sigmas)/sigmas_old))

    # Final data correction based on the final noise-covariance estimate
    W = np.diag(1/sigmas)
    WL = np.dot(W,LFM)
    WLLW = np.dot(WL,WL.T)
    # eq.(7)
    lamb = lambda0*np.trace(WLLW)/chanN
    P = np.dot(np.dot(WL.T, np.linalg.inv(WLLW + lamb*np.eye(chanN))), W)
    corrected_data = np.dot(np.dot(LFM, P), data)
    return corrected_data,P,sigmas

from numba import jit
@jit(nopython=True)
def _sound_numba(data, LFM, iter, lambda0=0.01):
    """Perform SOUND on one segment of data."""
    chanN = data.shape[0]
    sigmas = np.ones((chanN,1))

    # y_solved, sigmas = _DDWiener(data)
    # Compute the sample covariance matrix
    C = np.dot(data, data.T)
    # Compute the trace of the covariance matrix
    gamma = np.mean(np.diag(C))
    # Compute the DDWiener estimates in each channel
    y_solved = np.zeros_like(data)
    chanN = data.shape[0]
    for i in range(chanN):
        # idiff = np.setdiff1d(np.arange(chanN), i)
        idiff = np.array([j for j in np.arange(chanN) if j != i])
        # y_solved[i,:] = C[i,idiff] @ np.linalg.inv(C[idiff,:][:,idiff] + gamma * np.eye(chanN-1)) @ data[idiff,:]
        y_solved[i,:] = np.dot(np.dot(C[i,idiff], np.linalg.inv(C[idiff,:][:,idiff] + gamma * np.eye(chanN-1))),data[idiff,:])
    error = data-y_solved
    sigmas = np.sqrt(np.diag(np.dot(error,error.T)))/np.sqrt(data.shape[1])
    
    # Number of time points
    T = data.shape[1]
    # Going through all the channels as many times as requested
    dn = np.empty(shape=(0,))
    if isinstance(iter, int):
        for k in range(iter):
            # print("Iteration %s" % (k+1))s
            sigmas_old = sigmas.copy()
            # Evaluating each channel in a random order
            random_ch = np.random.permutation(chanN)
            for i in random_ch:
                
                chan =np.array([j for j in np.arange(chanN) if j != i])
                # Defining the whitening operator with the latest noise estimates
                W = np.diag(1/sigmas)
                # Computing the whitened version of the lead field
                WL = np.dot(W[chan,:][:,chan],LFM[chan,:])
                WLLW = np.dot(WL,WL.T)

                # Computing the MNE, the Wiener estimate in the studied channel
                # as well as the corresponding noise estimate
                # eq.(4)
                lamb = lambda0*np.trace(WLLW)/((chanN-1))
                # 类似于空间滤波器
                P = np.dot(np.dot(WL.T, np.linalg.inv(WLLW + lamb*np.eye(chanN-1))), W[chan,:][:,chan])
                
                # eq.(3)、eq.(6)
                y_solved = np.dot(np.dot(LFM, P), data[chan,:])
                # eq.(5)
                sigmas[i] = np.sqrt(np.sum((y_solved[i,:]-data[i,:])**2)/T)

            # Following and storing the convergence of the algorithm
            dn = np.append(dn,np.max(np.abs(sigmas_old - sigmas)/sigmas_old))

    elif isinstance(iter,str):
        k=0
        while True:
            if len(dn) > 0 and (dn[-1] < 0.01 or k == 100):
                break
            else:
                k = k + 1
                # print("Iteration %s" % (k))

            sigmas_old = sigmas.copy()
            # Evaluating each channel in a random order
            random_ch = np.random.permutation(chanN)
            for i in random_ch:
                
                # chan = np.setdiff1d(np.arange(chanN), i)
                chan = np.array([j for j in np.arange(chanN) if j != i])
                # Defining the whitening operator with the latest noise estimates
                W = np.diag(1/sigmas)
                # Computing the whitened version of the lead field
                WL = np.dot(W[chan,:][:,chan],LFM[chan,:])
                WLLW = np.dot(WL,WL.T)

                # Computing the MNE, the Wiener estimate in the studied channel
                # as well as the corresponding noise estimate
                # eq.(4)
                # if isinstance(lambda0,float):
                lamb = lambda0*np.trace(WLLW)/((chanN-1))
                # elif isinstance(lambda0,str):
                #     lamb = np.trace(WLLW)/np.trace(np.dot(y_solved,y_solved.T)/T)
                # print(lamb)
                # 类似于空间滤波器
                P = np.dot(np.dot(WL.T, np.linalg.inv(WLLW + lamb*np.eye(chanN-1))), W[chan,:][:,chan])
                
                # eq.(3)、eq.(6)
                y_solved = np.dot(np.dot(LFM, P), data[chan,:])
                # eq.(5)
                sigmas[i] = np.sqrt(np.sum((y_solved[i,:]-data[i,:])**2)/T)

            # Following and storing the convergence of the algorithm
            dn = np.append(dn,np.max(np.abs(sigmas_old - sigmas)/sigmas_old))

    # Final data correction based on the final noise-covariance estimate
    W = np.diag(1/sigmas)
    WL = np.dot(W,LFM)
    WLLW = np.dot(WL,WL.T)
    # eq.(7)
    lamb = lambda0*np.trace(WLLW)/chanN
    P = np.dot(np.dot(WL.T, np.linalg.inv(WLLW + lamb*np.eye(chanN))), W)
    corrected_data = np.dot(np.dot(LFM, P), data)
    return corrected_data,P,sigmas


@verbose
def repair_bad_channels(raw, LFM, iter='auto', lambda0=0.01,duration=None,
                        picks=None,use_numba=True,plot=False, verbose=None):
    """Denoise MEG channels using leave-one-out temporal projection.
    
    Parameters
    ----------
    raw : instance of Raw
        Raw data to denoise.
    LFM : np.array
        leadfield matrix.
    iter : int
        Number of iterations to perform.
    lambda0 : float
        Regularization parameter.
    duration(optional) : float | str
        The window duration (in seconds; default 10.) to use.
        If None, adaptive time series segmentation into chunks for cleaning.
    %(picks_all_data)s
    %(verbose)s

     Returns
    -------
    raw_twsound : instance of Raw
        The cleaned data.

    Notes
    -----
    This algorithm is computationally expensive, and can be several times
    slower than realtime for conventional M/EEG datasets.

    """
    
    print("Processing MEG data using TWSOUND...")
    picks = _picks_to_idx(raw.info, picks, exclude=())
    
    n_chans = len(picks)
    n_samples = raw.n_times
    sfreq = raw.info['sfreq']

    raw_twsound = raw.copy().load_data(verbose=False)

    if duration is None:
        print("Annotating bad segments...")
        raw_twsound = annotate_bad_segments(raw_twsound,plot=plot)
        
        chunk = []
        annotations = raw_twsound.annotations
        if verbose:
            print('find bad segments:',len(annotations))

        if len(annotations) != 0:
            
            for annotation in annotations:
                start = int(annotation['onset'] * raw_twsound.info["sfreq"]) - raw_twsound.first_samp
                chunk.append(start)
                
                stop = int((annotation['onset'] + annotation['duration']) * raw_twsound.info["sfreq"]) - raw_twsound.first_samp
                chunk.append(stop)

            if chunk[0] > 0:
                chunk.insert(0,0)
            if chunk[-1] < n_samples:
                chunk.append(n_samples)
            print("Using TWSOUN with ",len(chunk)-1," chunks")
            P = []
            sigmas = []
            for i in range(len(chunk)-1):
                print("Processing chunk ",i+1)
                start = chunk[i]
                stop = chunk[i+1]

                data_chunk = raw_twsound._data[picks,start:stop]

                U_lfm,S_lfm,Vh_lfm = scipy.linalg.svd(LFM,full_matrices=False)
                # LFM1 = copy.deepcopy(LFM)
                LFM1 = U_lfm @ np.diag(S_lfm)
                if use_numba:
                    data_clean,p,sigma = _sound_numba(data_chunk, LFM1, iter, lambda0)
                else:
                    data_clean,p,sigma = _sound(data_chunk, LFM1, iter, lambda0)
                raw_twsound._data[picks,start:stop] = data_clean
                P.append(p)
                sigmas.append(sigma)
            raw_twsound.set_annotations(mne.Annotations([], [], []))
            # return raw_twsound,P,sigmas
        else:
            print("No annotations found, processing all data")
            data = raw_twsound._data[picks,:]
            if use_numba:
                data_clean = _sound_numba(data, LFM, iter, lambda0)[0]
            else:
                data_clean = _sound(data, LFM, iter, lambda0)[0]
            raw_twsound._data[picks,:] = data_clean

        raw_twsound.set_annotations(mne.Annotations([], [], []))
        
    elif duration == 'all':
        print("Using TWSOUND with all data")
        data = raw_twsound._data[picks,:]
        if use_numba:
                data_clean = _sound_numba(data, LFM, iter, lambda0)[0]
        else:
            data_clean = _sound(data, LFM, iter, lambda0)[0]
        raw_twsound._data[picks,:] = data_clean
    
    else:
        print("Using TWSOUND with ",duration,"s")
        n_duration = int(round(float(duration) * sfreq))
        if n_duration < n_chans - 1:
            raise ValueError(
                "duration (%s) yielded %s samples, which is fewer "
                "than the number of channels -1 (%s)"
                % (n_duration / sfreq, n_duration, n_chans - 1)
            )
        n_overlap = n_duration // 2
        read_lims = list(range(0, n_samples, n_duration)) + [n_samples]

        s = _COLA(
            partial(_sound, LFM=LFM, iter=iter, lambda0=lambda0),
            _Storer(raw_twsound._data, picks=picks),
            n_samples,
            n_duration,
            n_overlap,
            sfreq,
        )
        for i,(start, stop) in tqdm(enumerate(zip(read_lims[:-1], read_lims[1:]))):
            s.feed(raw[picks, start:stop][0]) 
    
    return raw_twsound

def repairbads(raw,LFM,iter='auto', lambda0=0.01,duration=None,picks=None,
               return_raws=False,return_annotation_raw=False,
               use_numba=True,plot=False, verbose=None):
    
    if not isinstance(raw, (mne.io.RawArray,mne.io.Raw)):
        raise ValueError("raw must be of type mne.io.RawArray")
    
    print('------Repairing bad segments------')
    raw_reapired_segments,raw_annotation_first = repair_bad_segments(raw,
                                                                     return_annotation_raw=return_annotation_raw,
                                                                     plot=plot)

    print('------Repairing bad channels------')
    raw_reapired_channels = repair_bad_channels(raw_reapired_segments,LFM,iter=iter, 
                                                lambda0=lambda0,duration=duration,
                                                picks=picks,plot=plot,use_numba=use_numba,
                                                verbose=verbose)

    if return_annotation_raw and return_raws:
        return raw_annotation_first,raw_reapired_segments,raw_reapired_channels
    elif return_annotation_raw is True and return_raws is False:
        return raw_annotation_first,raw_reapired_channels
    elif return_annotation_raw is False and return_raws is True:
        return raw_reapired_segments,raw_reapired_channels
    else:
        return raw_reapired_channels
