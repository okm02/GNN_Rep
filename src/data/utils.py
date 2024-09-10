import torch
from dataclasses import dataclass, field
# typing
from typing import List, Dict, Any


@dataclass
class Node:
	# this class encodes the node of citeseer dataset
	feat_one_hot: list[int] = field(default_factory=list)
	label: int = -1
	connected_to: list[str] = field(default_factory=list)


def build_sparse_matrix(row_indices: List[int], col_indices: List[int]) -> torch.Tensor:
	"""
	:param row_indices: indices of rows in sparse matrix
	:param col_indices: indices of columns in sparse matrix
	:returns: sparse matrix
	"""
	# convert rows and columns into tensors
	row_tensor = torch.tensor(row_indices)
	col_tensor = torch.tensor(col_indices)
	value_tensor = torch.tensor([1] * len(row_indices))

	# build adjacency matrix
	csc = torch.sparse_csc_tensor(col_indices, row_indices, value_tensor, dtype=torch.float64)
	return csc

