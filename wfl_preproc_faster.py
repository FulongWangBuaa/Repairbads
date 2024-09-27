# %%
import scipy
import scipy.stats as sp_stats
import numpy as np
import mne



def _pearson_corr(x1, x2):
    """
    Computes Pearson's correlation coefficients between x1 and x2 sequences

    Parameters
    ----------
    x1 : numpy.ndarray
        Array with dimensions N1xL, where N1 is the number of sequences and
        L their length.
    x2 : numpy.ndarray
        Array with dimensions N2xL, where N2 is the number of sequences and
        L their length.

    Returns
    -------
    r : numpy.ndarray
        Array with dimensions N1xN2 containing the correlation coefficients
    """
    # Remove mean
    x1 = x1 - x1.mean(axis=0, keepdims=True)
    x2 = x2 - x2.mean(axis=0, keepdims=True)
    # Compute variance
    s1 = (x1**2).sum(axis=0, keepdims=True)
    s2 = (x2**2).sum(axis=0, keepdims=True)
    # Compute r
    return np.dot(x1, x2.T) / np.sqrt(np.dot(s1, s2.T))

def _var(x):
    """FASTER [Nolan2010]_ implementation of the variance.

    Parameters
    ----------
    x : numpy.ndarray
        Vector with the data sequence.

    Returns
    -------
    v : float
        Computed variance
    """
    if isinstance(x, list):
        return np.var(x)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            return np.var(x)
        elif x.ndim == 2:
            return np.var(x,axis=1)
        else:
            raise ValueError('Input array must be 1 or 2 dimensional')
    else:
        raise ValueError('Input array must be a list or numpy.ndarray')

def _hurst(x):
    """FASTER [Nolan2010]_ implementation of the Hurst Exponent.

    Parameters
    ----------
    x : numpy.ndarray
        Vector with the data sequence.

    Returns
    -------
    h : float
        Computed hurst exponent
    """

    # Get a copy of the data
    x0 = x.copy()
    x0_len = len(x)

    yvals = np.zeros(x0_len)
    xvals = np.zeros(x0_len)
    x1 = np.zeros(x0_len)

    index = 0
    binsize = 1

    while x0_len > 4:

        y = x0.std()
        index += 1
        xvals[index] = binsize
        yvals[index] = binsize*y

        x0_len = int(x0_len/2)
        binsize *= 2
        for ipoints in range(x0_len):
            x1[ipoints] = (x0[2*ipoints] + x0[2*ipoints - 1])*0.5

        x0 = x1[:x0_len]

    # First value is always 0
    xvals = xvals[1:index+1]
    yvals = yvals[1:index+1]

    logx = np.log(xvals)
    logy = np.log(yvals)

    p2 = np.polyfit(logx, logy, 1)
    return p2[0]

def _dX_dt(x):
    """
    Median time derivative value of the BSS components

    Parameters
    ----------
    bss_data : numpy.ndarray
        Array containing source signals. It must have dimensions CxTxE, where C
        is the number of components, T the number of time instants and E the
        number of events.

    Returns
    -------
    res : numpy.ndarray
        Vector of length C with the computed values for each component
    """
    if isinstance(x, list):
        return np.median(np.diff(x))
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            return np.median(np.diff(x))
        elif x.ndim == 2:
            return np.median(np.diff(x,axis=1),axis=1)
        else:
            raise ValueError('Input array must be 1 or 2 dimensional')
    else:
        raise ValueError('Input array must be a list or numpy.ndarray')

def _generate_color(mask):
    num_trials,num_chs = mask.shape
    colors = []
    for i in range(num_trials):
        trial_mask = mask[i,:]
        trial_color = []
        for j in range(num_chs):
            if trial_mask[j] == 0:
                trial_color.append('k')
            elif trial_mask[j] == 1:
                trial_color.append('b')
            elif trial_mask[j] == 2:
                trial_color.append('r')
            elif trial_mask[j] == 3:
                trial_color.append('lime')
        colors.append(trial_color)
    return colors

def FASTER(raw,tmin=-0.2,tmax=0.8,baseline=None,plot=False):
    #************************************Step 1********************************
    data_raw = raw.get_data(picks='data')*1e15
    ch_names = raw.copy().pick(picks='data').ch_names

    corr = []
    var = []
    hurst = []
    for ch_i,ch in enumerate(ch_names):
        data_ch = data_raw[ch_i,:]
        var.append(_var(data_ch))
        hurst.append(_hurst(data_ch))
        corr_ch = []
        ch_others = [c for c in ch_names if c != ch]
        for ch_other in ch_others:
            ch_j = ch_names.index(ch_other)
            corr_ch.append(_pearson_corr(data_ch,data_raw[ch_j,:]))
        corr.append(np.mean(corr_ch))

    corr_z = scipy.stats.zscore(corr)
    var_z = scipy.stats.zscore(var)
    hurst_z = scipy.stats.zscore(hurst)

    indices1 = {i for i in range(len(corr_z)) if corr_z[i] > 3}  
    indices2 = {i for i in range(len(var_z)) if var_z[i] > 3}  
    indices3 = {i for i in range(len(hurst_z)) if hurst_z[i] > 3}

    bad_ch_idx = list(indices1.union(indices2).union(indices3))

    raw1 = raw.copy()
    if len(bad_ch_idx) > 0:
        bad_ch = [ch_names[i] for i in bad_ch_idx]
        raw1.info['bads'] = bad_ch
        raw1.interpolate_bads()
    
    #************************************Step 2********************************
    stim_ch = raw1.copy().pick(picks='stim').ch_names[0]
    events = mne.find_events(raw1, stim_channel=stim_ch,verbose=False)
    epochs = mne.Epochs(raw1, events, tmin=tmin, tmax=tmax, baseline=baseline, preload=True,verbose=False)

    epochs_data = epochs.get_data(picks='data')*1e15
    num_trials1,num_chs,num_samples = epochs_data.shape
    trial_idx1 = list(np.arange(num_trials1))

    # 插值的坏道标记为1
    epoch_types = np.zeros((num_trials1,num_chs+1))
    if len(bad_ch_idx) > 0:
        epoch_types[:,bad_ch_idx] = np.ones((num_trials1,len(bad_ch_idx)))

    amp_range = []
    var = []
    deviation = []
    for trial_i in range(num_trials1):
        data_trial = epochs_data[trial_i,:,:]
        amp_range.append(np.mean(np.max(data_trial,axis=1) - np.min(data_trial,axis=1)))
        var.append(np.mean(_var(data_trial)))
        deviation_trial = np.mean(np.mean(data_trial,axis=1) - np.mean(data_raw,axis=1))
        deviation.append(deviation_trial)

    amp_range_z = scipy.stats.zscore(amp_range)
    var_z = scipy.stats.zscore(var)
    deviation_z = scipy.stats.zscore(deviation)

    indices1 = {i for i in range(len(amp_range_z)) if amp_range_z[i] > 3}  
    indices2 = {i for i in range(len(var_z)) if var_z[i] > 3}  
    indices3 = {i for i in range(len(deviation_z)) if deviation_z[i] > 3}

    bad_trial_idx = list(indices1.union(indices2).union(indices3))

    # 去除的epoch标记为2
    if len(bad_trial_idx) > 0:
        epoch_types[bad_trial_idx,:] = np.ones((len(bad_trial_idx),num_chs+1))*2

    epochs1 = epochs.copy()
    epochs1.drop(bad_trial_idx)

    trial_idxs2_in_1 = [i for i in trial_idx1 if i not in bad_trial_idx]


    #************************************Step 3********************************
    epochs2 = epochs1.copy()
    ch_names = epochs2.copy().pick(picks='data').ch_names
    epochs_data1 = epochs2.get_data(picks='data')*1e15
    num_trials2,num_chs,num_samples = epochs_data1.shape
    trial_idxs2 = list(np.arange(num_trials2))
    epochs_data1_ch_mean = np.mean(np.reshape(np.transpose(epochs_data1,(1,0,2)),(num_chs,num_trials2*num_samples)),axis=1)

    var = np.zeros((num_trials2,num_chs))
    median_slope = np.zeros((num_trials2,num_chs))
    amp_range = np.zeros((num_trials2,num_chs))
    deviation = np.zeros((num_trials2,num_chs))
    for trial_i in range(num_trials2):
        for ch_i in range(num_chs):
            data_trial_ch = epochs_data1[trial_i,ch_i,:]
            var_trial_ch = _var(data_trial_ch)
            median_slope_trial_ch = _dX_dt(data_trial_ch)
            amp_range_trial_ch = np.max(data_trial_ch) - np.min(data_trial_ch)
            deviation_trial_ch = np.mean(data_trial_ch) - epochs_data1_ch_mean[ch_i]
            var[trial_i,ch_i] = var_trial_ch
            median_slope[trial_i,ch_i] = median_slope_trial_ch
            amp_range[trial_i,ch_i] = amp_range_trial_ch
            deviation[trial_i,ch_i] = deviation_trial_ch

    var_z = scipy.stats.zscore(var)
    median_slope_z = scipy.stats.zscore(median_slope)
    amp_range_z = scipy.stats.zscore(amp_range)
    deviation_z = scipy.stats.zscore(deviation)

    var_bool = var_z > 3
    median_slope_bool = median_slope_z > 3
    amp_range_bool = amp_range_z > 3
    deviation_bool = deviation_z > 3

    combined_bool_alt = var_bool | median_slope_bool | amp_range_bool | deviation_bool

    bad_chs = []
    
    for trial_i in range(num_trials2):
        bool_trial = combined_bool_alt[trial_i,:]
        bad_chs_trial = [ch_names[i] for i in range(num_chs) if bool_trial[i]]
        bad_chs.append(bad_chs_trial)
        
        # 插值的epoch标记为3
        if len(bad_chs_trial) > 0:
            for ch_i in range(num_chs):
                if bool_trial[ch_i]:
                    epoch_types[trial_idxs2_in_1[trial_i],ch_i] = 3

            trial_idxs1 = trial_idxs2.copy()
            trial_idxs1.pop(trial_i)
            epochs_trial = epochs2.copy().drop(trial_idxs1,verbose=False)
            epochs_trial.info['bads'] = bad_chs_trial
            epochs_trial.interpolate_bads(verbose=False)
            epochs_trial_data = epochs_trial.get_data(copy=True)
            epochs2._data[trial_i,:,:] = epochs_trial_data

    
    if plot:
        epoch_colors = _generate_color(epoch_types)
        epochs.plot(epoch_colors=epoch_colors)

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        data = epoch_types.T
        rows, cols = data.shape
        color_map = {0: 'gray', 1: 'blue', 2: 'red', 3:'lime'}
        # 创建绘图
        fig, ax = plt.subplots(figsize=(13, 4),layout='constrained')  # 自适应大小
        # 绘制小矩形
        for i in range(rows):
            for j in range(cols):
                color = color_map[data[i, j]]  # 获取颜色
                # 创建小矩形
                rect = patches.Rectangle((j, rows - i - 1), 1, 1, color=color)
                ax.add_patch(rect)

        ax.axis('off')
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks(np.arange(0, cols, 1))
        ax.set_yticks(np.arange(0, rows, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()

    return epochs2,epoch_types

