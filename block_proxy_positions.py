"""Query positions shared by compact capture and offline block-proxy analysis."""

import torch


def tile_starts(sequence, size, fractions):
    count = sequence // size
    return sorted({
        min(count - 1, max(0, int(fraction * count))) * size
        for fraction in fractions
    })


def query_rows(start, size, sequence, count):
    end = min(start + size, sequence)
    if count >= end - start:
        return torch.arange(start, end)
    return torch.linspace(start, end - 1, count).round().long().unique()


def sampled_query_positions(sequence, sizes, fractions, rows_per_tile):
    positions = set()
    for size in sizes:
        for start in tile_starts(sequence, size, fractions):
            positions.update(
                query_rows(start, size, sequence, rows_per_tile).tolist()
            )
    return torch.tensor(sorted(positions), dtype=torch.long)
