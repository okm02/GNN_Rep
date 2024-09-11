from dataclasses import dataclass, field
# typing
from typing import List, Dict, Any


@dataclass
class Node:
	# this class encodes the node of citeseer dataset
	feat_one_hot: list[int] = field(default_factory=list)
	label: int = -1
	connected_to: list[str] = field(default_factory=list)

