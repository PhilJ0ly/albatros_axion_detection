# Philippe Joly 2025-06-18
# This script tests the fatstest pfb window functions (from standard C_contiguous input timestream)

import sys
import os
os.environ['NUMBA_OPT']='3'
os.environ['NUMBA_LOOP_VECTORIZE']='1'
os.environ['NUMBA_ENABLE_AVX']='1'
os.environ['NUMBA_CACHE_DIR'] = '/scratch/s/sievers/philj0ly/tmp/my_numba_cache'

import numpy as np
import numba as nb
# import scipy.fft as sfft
from scipy.fft import rfft, irfft, set_workers
import multiprocessing as mp
import psutil

sys.path.insert(0, os.path.expanduser("~"))
from albatros_analysis.src.utils.pfb_utils import *

# from albatros_analysis.src.utils.mkfftw import many_fft_r2c_1d as rfft, many_fft_c2r_1d as irfft

# from albatros_analysis.src.utils.mkfftw_float import many_fft_r2c_1d as rfft, many_fft_c2r_1d as irfft


def print_cpu_mem(str_msg):
    print(str_msg)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f'RSS Memory: {mem_info.rss / 1024**3:.2f} GB')
    print(f'VMS Memory: {mem_info.vms / 1024**3:.2f} GB')
    print(f'Available Memory: {psutil.virtual_memory().available / 1024**3:.2f} GB')


@nb.njit(parallel=True)
def nb_transpose_1d(x, out=None, cut=0):

    """
    Transposes a 2D array x in a parallelized, Numba-accelerated way.
    Creates an empty output array y, then fills it such that y[i, j] = x[j, i].
    Faster than x.T.copy()[cut:-cut].ravel() when used inside Numba-heavy workflows, especially with large matrices.
    """
    nr, nc = x.T.shape
    if out is None:
        out = np.empty((nr-2*cut)*nc,dtype=x.dtype)

    # RAVEL Output
    for i in nb.prange(nr-2*cut):
        for j in nb.prange(nc):
            out[i*nc+j]=x[j,i+cut]          
        
    return out


def compute_filter(matft, thresh):
    """
    Direct filter computation.
    
    Args:
        matft: Complex array
        thresh: Threshold value
    
    Returns:
        Filter array
    """
    filt = 1/np.conj(matft)
    if thresh>0:
        matft_abs_sq = np.abs(matft) ** 2
        thresh_sq = thresh * thresh
        filt *= matft_abs_sq / (thresh_sq + matft_abs_sq) * (1.0 + thresh_sq)
        
    return filt  


@nb.njit(parallel=True)
def apply_filter(ddft, filt):
    """
    Apply filter to ddft array in-place for memory efficiency.
    
    Args:
        ddft: Input/output array to be filtered
        filt: Filter array
    """
    for i in nb.prange(ddft.size):
        ddft.flat[i] *= filt.flat[i]

def cpu_ipfb(dat, out, filt, cut=0, n_workers=None):

    """
    CPU-based implementation of the inverse polyphase filterbank (iPFB).
    Performs:
        1. Inverse FFT on dat along axis 1 (assuming input is already in frequency domain).
        2. Transposes and applies forward FFT again.
        3. Applies an optional threshold filter.
        4. Final inverse FFT.
        5. Transposes the result back.

    Allows for CPU parallelism via scipy.fft.set_workers() and optional custom inverse matrix.
    """

    if n_workers is None:
        n_workers = mp.cpu_count()
    

    with set_workers(n_workers): 
        dd = irfft(dat, axis=1) # this one is acctually more efficient
        
        dd2 = dd.T

        ddft = rfft(dd2, axis=1) 
     
    
    apply_filter(ddft, filt)

    with set_workers(n_workers): 
        res = irfft(ddft , axis=1) 

    nb_transpose_1d(res, out=out, cut=cut)

@nb.njit(parallel=True)
def cpu_fpfb(timestream, win, scratch, lblock, ntap):

    """
    Core CPU function to compute the time-domain polyphase filterbank (PFB).
    Takes a flat timestream and a flat win (the filter), and computes
    filtered blocks using ntap taps of length lblock.
    Writes the results into scratch, a 2D array of shape (nblock, lblock).
    Highly optimized using Numba for JIT and parallelism.
    """

    nblock = timestream.size // lblock - (ntap - 1)
    
    for i in nb.prange(nblock):
        for k in range(lblock):
            scratch[i,k] = 0.0
            for j in range(ntap):
                scratch[i,k] += win[j*lblock + k] * timestream[i*lblock + j*lblock + k]


def cpu_pfb(timestream, win, scratch=None, nchan=2049, ntap=4, n_workers=None):

    """
    Full polyphase filterbank (PFB) pipeline on CPU.

    1. Computes filtered timestream using cpu_fpfb.
    2. Applies real FFT (rfft) along axis 1 to get frequency-domain representation.
    Returns the PFB spectrum.

    Supports multi-threading and reuse of pre-allocated memory (scratch, out).
    """
    

    lblock = 2 * (nchan - 1)
    nblock = timestream.size // lblock - (ntap - 1)

    if n_workers is None:
        n_workers = mp.cpu_count()

    if scratch is None:
        scratch = np.zeros((nblock, lblock), dtype=timestream.dtype)
    else:
        assert scratch.shape == (nblock, lblock)

    cpu_fpfb(timestream, win, scratch, lblock, ntap)

    with set_workers(n_workers):
        out=rfft(scratch,axis=1)

    return out


def get_matft_cpu(nslice,nchan=2049,ntap=4, n_workers=None):

    """
    1. Constructs the Fourier matrix (matft) used in the iPFB process on CPU.
    2. Creates a sinc-hamming window (dwin) over ntap taps and nchan channels.
    3. Fills the first ntap slices of mat with the window.
    4. Computes the FFT of the windowed matrix along axis 1.
    5. Returns the transformed matrix (matft).

    Intended to match nslice × nchan dimensionality for compatibility with iPFB.
    Assumes sinc_hamming is defined elsewhere in the module.
    """
    
    if n_workers is None:
        n_workers = mp.cpu_count()//2

    nn=2*(nchan-1)
    dwin=sinc_hamming(ntap,nn)

    cpu_win = np.asarray(dwin,dtype='float32',order='c')
    cpu_win= np.reshape(cpu_win,[ntap,len(cpu_win)//ntap])
    mat=np.zeros((nslice,nn),dtype='float32',order='c')
    mat[:ntap,:]=cpu_win
    
    mat=mat.T.copy()

    with set_workers(n_workers):
        matft = rfft(mat,axis=1)
    return matft


def test_pfb(niter=100):
    osamp =65536
    lblock=4096*osamp
    ntap=4
    N = lblock * ntap
    w = np.arange(0, N) - N // 2
    window='hamming'
    timestream = np.random.randn(1879048192).astype("float32")
    nblock = len(timestream)//lblock - (ntap - 1)
    win = np.__dict__[window](N) * np.sinc(w / lblock)
    scratch = np.empty((nblock, lblock), dtype="float32")
    # out = np.empty((nblock, 2048*osamp+1), dtype='complex64',order='C')

        
    # PFB 
    print(f"Testing PFB Speeds on {timestream.size} Timestream with {niter} Iterations")

    # warm up
    for i in range(min(2, niter//10)):
        y = cpu_pfb(timestream, win, scratch=scratch, nchan=lblock//2+1, ntap=ntap)
    print("Warmed Up")
    tot = 0
    for i in range(niter):
        t1=time.time()
        y=cpu_pfb(timestream, win, scratch=scratch, nchan=lblock//2+1, ntap=ntap)
        t2=time.time()
        tot+=t2-t1
        if i%(niter//10)==0:
            print(i, tot/(i+1))

    tot /= niter
    print("\t", tot, timestream.nbytes/tot/(1e6))


# dat, matft, out, inv_matft=None, thresh=0.0, cut=0, n_workers=None
def test_ipfb(niter = 100):
    to_ipfbr = np.random.randn(524288*2049).reshape(524288, 2049).astype("float32")
    to_ipfbi = np.random.randn(524288*2049).reshape(524288, 2049).astype("float32")
    to_ipfb = to_ipfbr+1j*to_ipfbi

    out = np.empty(1879048192, dtype="float32")
    matft = get_matft_cpu(524288)
    filt = compute_filter(matft, 0.45)

    
    print(f"Testing IPFB Speeds on {to_ipfb.shape} Input with {niter} Iterations")

    # warm up
    for i in range(min(4, niter//10)):
        y = cpu_ipfb(to_ipfb, out, filt, cut=32768)
    print("Warmed Up")
    tot = 0
    for i in range(niter):
        t1=time.time()
        cpu_ipfb(to_ipfb, out, filt, cut=32768)
        t2=time.time()
        tot+=t2-t1
        if i%(niter//10)==0:
            print(i, tot/(i+1))

    tot /= niter
    print("\t", tot, "s", to_ipfb.nbytes/tot/(1e6), "MSPS")

        
if __name__=='__main__':
    test_pfb(100)