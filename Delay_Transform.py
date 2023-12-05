'''Some functions for the Fourier Transform of visibility data, including the application of a windowing function.
    Taken from https://github.com/HERA-Team/uvtools, specifically
    
    https://github.com/HERA-Team/uvtools/blob/main/uvtools/utils.py 
    https://github.com/HERA-Team/uvtools/blob/main/uvtools/dspec.py'''

import numpy as np
from scipy.signal import windows
from numpy.fft import fft, fftshift, ifftshift
from astropy import units

def gen_window(window, N, alpha=0.5, edgecut_low=0, edgecut_hi=0, normalization=None, **kwargs):
    
    """
    Generate a 1D window function of length N.

    Args:
        window : str, window function
        N : int, number of channels for windowing function.
        edgecut_low : int, number of bins to consider as zero-padded at the low-side
            of the array, such that the window smoothly connects to zero.
        edgecut_hi : int, number of bins to consider as zero-padded at the high-side
            of the array, such that the window smoothly connects to zero.
        alpha : if window is 'tukey', this is its alpha parameter.
        normalization : str, optional
            set to 'rms' to divide by rms and 'mean' to divide by mean.
    """
    if normalization is not None:
        if normalization not in ["mean", "rms"]:
            raise ValueError("normalization must be one of ['rms', 'mean']")
    # parse multiple input window or special windows
    w = np.zeros(N, dtype=float)
    Ncut = edgecut_low + edgecut_hi
    if Ncut >= N:
        raise ValueError("Ncut >= N for edgecut_low {} and edgecut_hi {}".format(edgecut_low, edgecut_hi))
    if edgecut_hi > 0:
        edgecut_hi = -edgecut_hi
    else:
        edgecut_hi = None
    if window in ['none', None, 'None', 'boxcar', 'tophat']:
        w[edgecut_low:edgecut_hi] = windows.boxcar(N - Ncut)
    elif window in ['blackmanharris', 'blackman-harris', 'bh', 'bh4']:
        w[edgecut_low:edgecut_hi] =  windows.blackmanharris(N - Ncut)
    elif window in ['hanning', 'hann']:
        w[edgecut_low:edgecut_hi] =  windows.hann(N - Ncut)
    elif window == 'tukey':
        w[edgecut_low:edgecut_hi] =  windows.tukey(N - Ncut, alpha)
    elif window in ['blackmanharris-7term', 'blackman-harris-7term', 'bh7']:
        # https://ieeexplore.ieee.org/document/293419
        a_k = [0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837,
              0.00077658482522, 0.00001388721735]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-9term', 'cosinesum9term', 'cs9']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.384331152777942e-1, 4.00554534864382e-1, 2.358242530472107e-1, 9.527918858383112e-2,
               2.537395516617152e-2, 4.152432907505835e-3, 3.68560416329818e-4, 1.38435559391703e-5,
               1.161808358932861e-7]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    elif window in ['cosinesum-11term', 'cosinesum11term', 'cs11']:
        # https://ieeexplore.ieee.org/document/940309
        a_k = [2.151527506679809e-1, 3.731348357785249e-1, 2.424243358446660e-1, 1.166907592689211e-1,
               4.077422105878731e-2, 1.000904500852923e-2, 1.639806917362033e-3, 1.651660820997142e-4,
               8.884663168541479e-6, 1.938617116029048e-7, 8.482485599330470e-10]
        w[edgecut_low:edgecut_hi] = windows.general_cosine(N - Ncut, a_k, True)
    else:
        try:
            # return any single-arg window from windows
            w[edgecut_low:edgecut_hi] = getattr(windows, window)(N - Ncut)
        except AttributeError:
            raise ValueError("Didn't recognize window {}".format(window))
    if normalization == 'rms':
        w /= np.sqrt(np.mean(np.abs(w)**2.))
    if normalization == 'mean':
        w /= w.mean()
    return w

def fourier_freqs(times):
    
    """A function for generating Fourier frequencies given 'times'.

    Parameters
    ----------
    times : np.ndarray, shape=(Ntimes,)
        An array of parameter values. These are nominally referred to as times,
        but may be frequencies or other parameters for which a Fourier dual can
        be defined. This function assumes a uniform sampling rate.

    Returns
    -------
    freqs : np.ndarray, shape=(Ntimes,)
        An array of coordinates dual to the input coordinates. Similar to the
        output of np.fft.fftfreq, but ordered so that the zero frequency is in
        the center of the array.
    """
    

    # get the number of samples and the sample rate
    N = len(times)
    dt = np.mean(np.diff(times))

    # get the Nyquist frequency
    f_nyq = 1.0 / (2 * dt)

    # return the frequency array
    return np.linspace(-f_nyq, f_nyq, N, endpoint=False)

def FFT(data, axis=-1, taper=None, **kwargs):
    
    """Convenient function for performing a FFT along an axis.

    Parameters
    ----------
    data : np.ndarray
       An array of data, assumed to not be ordered in the numpy FFT convention.
       Typically the data_array of a UVData object.

    axis : int, optional
        The axis to perform the FFT over. Default is the last axis of the array.

    taper : str, optional
        Choice of taper (windowing function) to apply to the data. Default is 
        to use no taper.

    **kwargs
        Keyword arguments for generating the taper. Passed directly to 
        ``uvtools.dspec.gen_window``.

    Returns
    -------
    data_fft : np.ndarray
        The Fourier transform of the data along the specified axis. The array
        has the same shape as the original data, with the same ordering.
    """
    # Ensure that the trick for reshaping the window works for 1-D arrays.
    axis = np.arange(data.ndim)[axis]
    window = gen_window(taper, data.shape[axis], **kwargs)
    new_shape = tuple([1 if ax != axis else -1 for ax in range(data.ndim)])
    window.shape = new_shape

    # It's not obvious what the correct way to shift the data is; numpy docs
    # state that fft(A) gives the Fourier transform, with correct phases and
    # amplitudes, with the convention that positive frequencies are first,
    # followed by negative frequencies (i.e. fftshift(fft(A)) gives the right
    # answer ordered to be monotonically increasing in Fourier frequency).
    # Contrary to that, there are discussions on stackoverflow that suggest
    # the phases will be incorrect *unless* the data is subjected to an
    # ifftshift before being transformed; however, some simple tests suggest
    # that shifting the data prior to transforming actually creates phase
    # instabilities and conjugates the transformed data relative to what is
    # expected. So we perform no shifting prior to transforming.
    return fftshift(fft(window * data, axis=axis), axis)

