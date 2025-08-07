
# Philippe Joly 2025-06-26

""" 
GPU RePFB Stream - Refactored for Readability

This script implements a RePFB algorithm to change the frequency resolution
of ALBATROS telescope data with improved modularity and readability.
"""

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))
from albatros_analysis.src.correlations import baseband_data_classes as bdc
from albatros_analysis.src.correlations import correlations as cr
import cupy as cp
from albatros_analysis.src.utils import pfb_gpu_utils as pu
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import ctypes
lib = ctypes.CDLL('./libcgemm_batch.so')

lib.cgemm_strided_batched.argtypes = [
    ctypes.c_void_p,  # A.ptr
    ctypes.c_void_p,  # B.ptr
    ctypes.c_void_p,  # C.ptr
    ctypes.c_int,     # M
    ctypes.c_int,     # N
    ctypes.c_int,     # K
    ctypes.c_int      # batchCount
]
lib.cgemm_strided_batched.restype = None


@dataclass
class ProcessingConfig:
    """Configuration for RePFB processing"""
    acclen: int  # Accumulation length in units of 4096-sample IPFB output blocks
    pfb_size: int  # Size of the PFB transform
    nchunks: int  # Total number of IPFB output chunks to process
    nblock: int  # Number of PFB blocks per iteration (streamed)
    chanstart: int  # Start frequency channel
    chanend: int  # End frequency channel
    osamp: int  # Oversampling factor for RePFB
    cut: int = 10  # Rows to cut from top/bottom of IPFB
    filt_thresh: float = 0.45  # Regularization parameter for IPFB deconvolution
    ntap: int = 4  # Number of taps
    npol: int = 2  # Number of polarizations


@dataclass
class BufferSizes:
    """Calculated buffer sizes"""
    lblock: int  # Length of each block
    szblock: int  # Size of streaming block buffer
    lchunk: int  # Length of chunk after IPFB
    nchan: int  # Number of channels after oversampling
    
    @classmethod
    def from_config(cls, config: ProcessingConfig) -> 'BufferSizes':
        lblock = 4096 * config.osamp
        szblock = int((config.nblock + (config.ntap - 1)) * lblock)
        lchunk = 4096 * config.acclen
        nchan = (config.chanend - config.chanstart) * config.osamp
        return cls(lblock, szblock, lchunk, nchan)


class BufferManager:
    """Manages GPU buffers for streaming processing"""
    
    def __init__(self, config: ProcessingConfig, sizes: BufferSizes, nant: int):
        self.config = config
        self.sizes = sizes
        self.nant = nant
        
        # Initialize buffers
        self.pol = cp.empty((config.acclen + 2*config.cut, 2049), dtype='complex64', order='C')
        self.cut_pol = cp.zeros((nant, config.npol, 2*config.cut, 2049), dtype='complex64', order='C')
        self.pfb_buf = cp.zeros((nant, config.npol, config.nblock + (config.ntap-1), sizes.lblock), 
                               dtype='float32', order='C')
        self.rem_buf = cp.empty((nant, config.npol, sizes.lchunk), dtype='float32', order='C')
        
        # Buffer indices
        self.rem_idx = np.zeros((nant, config.npol), dtype=np.uint64)
        self.pfb_idx = np.zeros((nant, config.npol), dtype=np.uint64)
    
    def reset_overlap_region(self):
        """Set up overlap region for continuity between iterations"""
        if hasattr(self, '_first_iteration') and not self._first_iteration:
            # Keep overlap from previous iteration
            ntap = self.config.ntap
            self.pfb_buf[:, :, :ntap-1, :] = self.pfb_buf[:, :, -(ntap-1):, :]
            self.pfb_idx[:, :] = (ntap - 1) * self.sizes.lblock
        else:
            self._first_iteration = False
    
    def add_remaining_to_pfb_buffer(self, ant_idx: int, pol_idx: int) -> bool:
        """
        Add remaining buffer data to PFB buffer.
        Returns True if buffer is full, False if more data needed.
        """
        rem_size = self.rem_idx[ant_idx, pol_idx]
        pfb_pos = self.pfb_idx[ant_idx, pol_idx]
        
        if pfb_pos + rem_size > self.sizes.szblock:
            # Need to handle overflow
            available_space = self.sizes.szblock - pfb_pos
            self._handle_buffer_overflow(ant_idx, pol_idx, available_space)
            return True
        else:
            # Normal case: add all remaining data
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:pfb_pos + rem_size] = \
                self.rem_buf[ant_idx, pol_idx, :rem_size]
            self.pfb_idx[ant_idx, pol_idx] += rem_size
            self.rem_idx[ant_idx, pol_idx] = 0
            return False
    
    def _handle_buffer_overflow(self, ant_idx: int, pol_idx: int, available_space: int):
        """Handle case where remaining buffer exceeds available PFB space"""
        rem_size = self.rem_idx[ant_idx, pol_idx]
        
        # Fill remaining PFB space
        self.pfb_buf[ant_idx, pol_idx].flat[self.pfb_idx[ant_idx, pol_idx]:] = \
            self.rem_buf[ant_idx, pol_idx, :available_space]
        
        # Shift remaining data
        overflow_size = rem_size - available_space
        self.rem_buf[ant_idx, pol_idx, :overflow_size] = \
            self.rem_buf[ant_idx, pol_idx, available_space:rem_size]
        
        self.rem_idx[ant_idx, pol_idx] = overflow_size
        self.pfb_idx[ant_idx, pol_idx] = self.sizes.szblock
    
    def add_chunk_to_buffer(self, ant_idx: int, pol_idx: int, processed_chunk: cp.ndarray) -> bool:
        """
        Add processed chunk to buffers.
        Returns True if PFB buffer is full.
        """
        chunk_size = len(processed_chunk)
        pfb_pos = self.pfb_idx[ant_idx, pol_idx]
        final_pos = pfb_pos + chunk_size
        
        if final_pos > self.sizes.szblock:
            # Handle overflow
            available_space = self.sizes.szblock - pfb_pos
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:] = processed_chunk[:available_space]
            
            # Store overflow in remaining buffer
            overflow_size = chunk_size - available_space
            self.rem_buf[ant_idx, pol_idx, :overflow_size] = processed_chunk[available_space:]
            self.rem_idx[ant_idx, pol_idx] = overflow_size
            self.pfb_idx[ant_idx, pol_idx] = self.sizes.szblock
            return True
        else:
            # Normal case
            self.pfb_buf[ant_idx, pol_idx].flat[pfb_pos:final_pos] = processed_chunk
            self.pfb_idx[ant_idx, pol_idx] = final_pos
            return final_pos == self.sizes.szblock
    
    def pad_incomplete_buffer(self, ant_idx: int, pol_idx: int):
        """Pad incomplete PFB buffer with zeros"""
        current_pos = self.pfb_idx[ant_idx, pol_idx]
        if current_pos < self.sizes.szblock:
            self.pfb_buf[ant_idx, pol_idx].flat[current_pos:] = 0.0


class IPFBProcessor:
    """Handles inverse PFB processing of chunks"""
    
    def __init__(self, config: ProcessingConfig, channels: np.ndarray, filt: cp.ndarray):
        self.config = config
        self.channels = channels
        self.filt = filt
    
    def process_chunk(self, chunk: dict, pol_idx: int, start_specnum: int, cut_pol_data: cp.ndarray) -> cp.ndarray:
        """Process a single chunk through inverse PFB"""
        # Prepare polarization data with edge handling
        pol_data = cp.empty((self.config.acclen + 2*self.config.cut, 2049), dtype='complex64', order='C')
        pol_data[:2*self.config.cut] = cut_pol_data
        
        # Make continuous data
        continuous_data = bdc.make_continuous_gpu(
            chunk[f"pol{pol_idx}"],
            chunk['specnums'],
            start_specnum,
            self.channels[slice(None)],  # Use all channels
            self.config.acclen,
            nchans=2049
        )
        pol_data[2*self.config.cut:] = continuous_data
        
        # Apply inverse PFB and remove edge effects
        result = pu.cupy_ipfb(pol_data, self.filt)[self.config.cut:-self.config.cut]
        return result.ravel()


class MissingDataTracker:
    """Tracks missing data fractions across processing"""
    
    def __init__(self, nant: int, n_job_chunks: int):
        self.missing_count = [[] for _ in range(nant)]
        self.missing_fraction = np.zeros((nant, n_job_chunks), dtype='float64')
    
    def add_chunk_info(self, ant_idx: int, chunk: dict, acclen: int, pfb_blocks_affected: int):
        """Add missing data info for a chunk"""
        missing_pct = (1 - len(chunk["specnums"]) / acclen) * 100
        self.missing_count[ant_idx].append([missing_pct, pfb_blocks_affected])
    
    def calculate_job_average(self, ant_idx: int, job_idx: int):
        """Calculate average missing fraction for a job"""
        total_missing = 0.0
        count = 0
        
        for miss_info in self.missing_count[ant_idx]:
            total_missing += miss_info[0]
            miss_info[1] -= 1  # Decrement blocks affected
            count += 1
        
        if count > 0:
            self.missing_fraction[ant_idx, job_idx] = total_missing / count
        
        # Remove entries that won't affect future jobs
        self.missing_count[ant_idx] = [info for info in self.missing_count[ant_idx] if info[1] > 0]


class JobChunkPlanner:
    """Plans how to chunk jobs for streaming processing"""
    
    @staticmethod
    def plan_chunks(nchunks: int, lchunk: int, nblock: int, lblock: int, ntap: int) -> List[Tuple[int, Optional[int]]]:
        """
        Plan job chunks to efficiently use buffers.
        Returns list of (start_chunk, end_chunk) tuples.
        """
        ranges = []
        stride_size = nblock * lblock
        overlap_size = (ntap - 1) * lblock
        total_needed = stride_size + overlap_size
        
        remainder = 0
        cur_chunk = 0
        
        while cur_chunk < nchunks:
            start_chunk = cur_chunk
            
            # Accumulate chunks until we have enough samples
            while remainder < total_needed and cur_chunk < nchunks:
                remainder += lchunk
                cur_chunk += 1
            
            if remainder >= total_needed:
                ranges.append((start_chunk, cur_chunk))
                remainder -= stride_size
            else:
                # Handle final incomplete chunk
                if remainder > 0:
                    ranges.append((start_chunk, cur_chunk))
                break
        
        return ranges


def setup_antenna_objects(idxs: List[int], files: List[str], config: ProcessingConfig) -> Tuple[List, np.ndarray]:
    """Set up antenna file iterators"""
    antenna_objs = []
    for i, (idx, file) in enumerate(zip(idxs, files)):
        antenna = bdc.BasebandFileIterator(
            file, 0, idx, config.acclen, nchunks=config.nchunks,
            chanstart=config.chanstart, chanend=config.chanend, type='float'
        )
        antenna_objs.append(antenna)
    
    # Get channel information from first antenna
    channels = np.asarray(antenna_objs[0].obj.channels, dtype='int64')
    return antenna_objs, channels


def setup_filters_and_windows(config: ProcessingConfig, cupy_win_big: Optional[cp.ndarray] = None, 
                             filt: Optional[cp.ndarray] = None) -> Tuple[cp.ndarray, cp.ndarray]:
    """Set up PFB window and IPFB filter"""
    if cupy_win_big is None:
        dwin = pu.sinc_hamming(config.ntap, 4096 * config.osamp)
        cupy_win_big = cp.asarray(dwin, dtype='float32', order='c')
    
    if filt is None:
        matft = pu.get_matft(config.acclen + 2*config.cut)
        filt = pu.calculate_filter(matft, config.filt_thresh)
    
    return cupy_win_big, filt


def repfb_xcorr_avg(idxs: List[int], files: List[str], acclen: int, nchunks: int, nblock: int, 
                   chanstart: int, chanend: int, osamp: int, cut: int = 10, filt_thresh: float = 0.45,
                   cupy_win_big: Optional[cp.ndarray] = None, filt: Optional[cp.ndarray] = None, 
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, cp.ndarray, cp.ndarray]:
    """
    Perform reverse PFB and GPU-based cross-correlation on streaming baseband data.
    
    Args:
        idxs: List of antenna indices
        files: List of baseband files per antenna
        acclen: Accumulation length in units of 4096-sample IPFB output blocks
        nchunks: Total number of IPFB output chunks to process
        nblock: Number of PFB blocks per iteration (streamed)
        chanstart, chanend: Frequency channel bounds
        osamp: Oversampling factor for RePFB
        cut: Number of rows to cut from top and bottom of IPFB
        filt_thresh: Regularization parameter for IPFB deconvolution filter
        cupy_win_big: Precomputed PFB window (optional)
        filt: Precomputed IPFB filter (optional)
        verbose: Enable verbose logging
    
    Returns:
        vis: Averaged cross-correlation matrix per time and frequency
        missing_fraction: Fraction of missing data per chunk and antenna
        freqs: Array of final frequency bins processed
        cupy_win_big: PFB window used
        filt: IPFB filter used
    """
    # Setup configuration
    config = ProcessingConfig(
        acclen=acclen, pfb_size=0, nchunks=nchunks, nblock=nblock,
        chanstart=chanstart, chanend=chanend, osamp=osamp, cut=cut, filt_thresh=filt_thresh
    )
    sizes = BufferSizes.from_config(config)
    nant = len(idxs)
    
    # Setup components
    antenna_objs, channels = setup_antenna_objects(idxs, files, config)
    cupy_win_big, filt = setup_filters_and_windows(config, cupy_win_big, filt)
    
    # Initialize managers
    buffer_mgr = BufferManager(config, sizes, nant)
    ipfb_processor = IPFBProcessor(config, channels, filt)
    job_chunks = JobChunkPlanner.plan_chunks(nchunks, sizes.lchunk, nblock, sizes.lblock, config.ntap)
    missing_tracker = MissingDataTracker(nant, len(job_chunks))
    
    # Setup frequency ranges
    repfb_chanstart = channels[config.chanstart] * osamp
    repfb_chanend = channels[config.chanend] * osamp
    
    # Initialize output arrays
    xin = cp.empty((nant * config.npol, nblock, sizes.nchan), dtype='complex64', order='F')
    vis = np.zeros((nant * config.npol, nant * config.npol, sizes.nchan, len(job_chunks)), 
                   dtype="complex64", order="F")
    
    start_specnums = [ant.spec_num_start for ant in antenna_objs]
    jobs = list(zip(*antenna_objs))
    
    if verbose:
        _print_debug_info(config, sizes, buffer_mgr, len(job_chunks), nchunks)
    
    # Main processing loop
    total_time = 0.0
    for job_idx, (chunk_start, chunk_end) in enumerate(job_chunks):
        start_time = time.perf_counter()
        
        buffer_mgr.reset_overlap_region()
        subjobs = jobs[chunk_start:chunk_end] if chunk_end is not None else jobs[chunk_start:]
        
        # Process each antenna and polarization
        for ant_idx in range(nant):
            for pol_idx in range(config.npol):
                _process_antenna_pol(
                    ant_idx, pol_idx, subjobs, buffer_mgr, ipfb_processor,
                    missing_tracker, job_idx, start_specnums[ant_idx], 
                    channels, config, sizes, verbose
                )
                
                # Generate PFB output for cross-correlation
                output_start = ant_idx * config.npol + pol_idx
                xin[output_start, :, :] = pu.cupy_pfb(
                    buffer_mgr.pfb_buf[ant_idx, pol_idx], 
                    cupy_win_big,
                    nchan=2048 * osamp + 1, 
                    ntap=4
                )[:, repfb_chanstart:repfb_chanend]
        
        # Compute cross-correlations
        vis[:, :, :, job_idx] = cp.asnumpy(
            cr.avg_xcorr_all_ant_gpu(xin, nant, config.npol, nblock, sizes.nchan, split=1)
        )
        
        end_time = time.perf_counter()
        total_time += end_time - start_time
        
        if verbose and job_idx % 100 == 0:
            print(f"Job Chunk {job_idx + 1}/{len(job_chunks)}, avg time {total_time/(job_idx + 1):.4f} s")
    
    if verbose:
        print("=" * 30)
        print(f"Completed {len(job_chunks)}/{len(job_chunks)} Job Chunks")
        print(f"avg time per job: {total_time/len(job_chunks):.4f} s")
        print("=" * 30)
    
    vis = np.ma.masked_invalid(vis)
    freqs = np.arange(repfb_chanstart, repfb_chanend)
    
    return vis, missing_tracker.missing_fraction, freqs, cupy_win_big, filt


def _process_antenna_pol(ant_idx: int, pol_idx: int, subjobs: List, buffer_mgr: BufferManager,
                        ipfb_processor: IPFBProcessor, missing_tracker: MissingDataTracker,
                        job_idx: int, start_specnum: int, channels: np.ndarray, 
                        config: ProcessingConfig, sizes: BufferSizes, verbose: bool):
    """Process a single antenna/polarization combination"""
    
    # Handle any remaining data from previous iteration
    buffer_full = buffer_mgr.add_remaining_to_pfb_buffer(ant_idx, pol_idx)
    
    if buffer_full:
        return  # Buffer already full, nothing more to do
    
    # Process each chunk for this antenna/polarization
    for chunk_idx, chunks in enumerate(subjobs):
        chunk = chunks[ant_idx]
        
        # Track missing data (only for first polarization to avoid duplication)
        if pol_idx == 0:
            pfb_blocks_affected = ((buffer_mgr.pfb_idx[ant_idx, pol_idx] + sizes.lchunk) // sizes.lblock) % config.nblock + 1
            missing_tracker.add_chunk_info(ant_idx, chunk, config.acclen, pfb_blocks_affected)
        
        # Process chunk through inverse PFB
        processed_chunk = ipfb_processor.process_chunk(
            chunk, pol_idx, start_specnum, 
            buffer_mgr.cut_pol[ant_idx, pol_idx, :, :]
        )
        
        # Update edge data for next iteration
        pol_data = cp.empty((config.acclen + 2*config.cut, 2049), dtype='complex64', order='C')
        pol_data[2*config.cut:] = bdc.make_continuous_gpu(
            chunk[f"pol{pol_idx}"], chunk['specnums'], start_specnum,
            channels[slice(None)], config.acclen, nchans=2049
        )
        buffer_mgr.cut_pol[ant_idx, pol_idx, :, :] = pol_data[-2*config.cut:]
        
        # Add processed chunk to buffer
        buffer_full = buffer_mgr.add_chunk_to_buffer(ant_idx, pol_idx, processed_chunk)
        
        if buffer_full:
            break
    
    # Calculate missing data average for this job
    if pol_idx == 0:
        missing_tracker.calculate_job_average(ant_idx, job_idx)
    
    # Pad buffer if incomplete
    if buffer_mgr.pfb_idx[ant_idx, pol_idx] != sizes.szblock:
        if verbose:
            print(f"Job {job_idx + 1} (Ant {ant_idx}, pol {pol_idx}): "
                  f"Incomplete pfb_buffer with only {buffer_mgr.pfb_idx[ant_idx, pol_idx]} "
                  f"instead of {sizes.szblock}")
        buffer_mgr.pad_incomplete_buffer(ant_idx, pol_idx)


def _print_debug_info(config: ProcessingConfig, sizes: BufferSizes, buffer_mgr: BufferManager, 
                     n_job_chunks: int, nchunks: int):
    """Print debug information if verbose mode is enabled"""
    print(f"nblock: {config.nblock}, lblock: {sizes.lblock}")
    print(f"ipfb_size: {config.acclen}")
    print(f"window shape: {buffer_mgr.pfb_buf.shape}")
    print(f"filter shape: Not shown")  # filt shape would need to be passed
    print(f"pol: {buffer_mgr.pol.shape}")
    print(f"pfb_buf: {buffer_mgr.pfb_buf.shape}, rem_buf: {buffer_mgr.rem_buf.shape}")
    print(f"chunky inputs: {nchunks}, {sizes.lchunk}, {config.nblock}, {sizes.lblock}, {config.ntap}")
    print(f"starting {n_job_chunks} PFB Jobs over {nchunks} IPFB chunks...")
    pu.print_mem("START ITER")