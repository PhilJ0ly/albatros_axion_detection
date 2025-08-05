import numpy as np


def get_chunkier(nchunks, lchunk, nblock, lblock, ntap):
    ranges = []
    stride_sz = nblock * lblock
    overlap_sz = (ntap - 1) * lblock
    total_needed = stride_sz + overlap_sz

    remainder = 0
    cur_chunk = 0
    sample_offset = 0  # total number of samples processed so far

    while True:
        start_chunk = cur_chunk
        added = 0

        # Accumulate until we have enough samples
        while remainder < total_needed and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
            added += 1

        if remainder >= total_needed:
            ranges.append((start_chunk, cur_chunk))
            remainder -= stride_sz  # we retain the overlap
            sample_offset += stride_sz
        else:
            if remainder > 0:
                ranges.append((start_chunk, None))
            break

    return ranges

def get_streaming_chunks(nchunks, lchunk, nblock, lblock, overlap_factor=0.5):
    """Generate chunks with proper overlap for streaming PFB"""
    shift_size = int(nblock * (1 - overlap_factor))
    chunks_per_buffer = (nblock * lblock) // lchunk
    
    ranges = []
    for start_chunk in range(0, nchunks - chunks_per_buffer + 1, 
                           max(1, int(chunks_per_buffer * (1 - overlap_factor)))):
        end_chunk = min(start_chunk + chunks_per_buffer, nchunks)
        ranges.append((start_chunk, end_chunk))
    
    return ranges

def get_chunky(nchunks, lchunk, nblock, lblock):
    ranges = []
    big_sz = nblock * lblock
    need = big_sz
    remainder = 0
    cur_chunk = 0

    while True:
        start_chunk = cur_chunk
        used = 0

        while remainder < need and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
            used += 1

        if remainder >= need:
            ranges.append((start_chunk, cur_chunk))
            remainder -= need
        else:
            # Not enough chunks left to satisfy need
            ranges.append((start_chunk, None))
            break

        # After the first full fill, just fill 1 block at a time
        need = lblock

        # Also handle the case where we can satisfy `need` from remainder only
        if remainder >= need and used == 0:
            # We just use remainder — no chunk consumed
            ranges.append((cur_chunk, cur_chunk))
            remainder -= need
            need = lblock

    return ranges

def get_chunky_overlap(nchunks, lchunk, nblock, lblock, overlap_factor=0.25):
    ranges = []
    big_sz = nblock * lblock
    overlap = int(overlap_factor * big_sz)
    need = big_sz

    remainder = 0
    cur_chunk = 0

    while True:
        start_chunk = cur_chunk
        used = 0
        new_data = 0

        while new_data < need and cur_chunk < nchunks:
            remainder += lchunk
            cur_chunk += 1
            used += 1
            new_data += lchunk

        if remainder >= need:
            ranges.append((start_chunk, cur_chunk))
            remainder -= need
        else:
            ranges.append((start_chunk, None))
            break

        need = big_sz - overlap

        while remainder >= need:
            ranges.append((cur_chunk, cur_chunk))
            remainder -= need
    return ranges

if __name__ == "__main__":
    # print(get_streaming_chunks(21,100,4,100,0.75))
    print(get_chunky(21,100,4,100))
    # print(get_chunky_overlap(21,100,4,100,0.75))
    print(get_chunkier(21, 100, 5, 90, 4))