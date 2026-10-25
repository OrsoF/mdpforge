"""Shared traversal of sparse transition matrices."""

from scipy.sparse import csr_matrix


def _nonzero_entries(matrix: csr_matrix):
    """Traverse stored CSR entries, preserving coordinate sums for duplicates."""
    entries = matrix.tocoo()
    canonical = matrix.has_canonical_format
    for row, column, value in zip(entries.row, entries.col, entries.data):
        if value != 0:
            if not canonical:
                value = matrix[row, column]
            yield int(row), int(column), float(value)
