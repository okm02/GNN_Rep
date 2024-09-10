import os, torch, argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
# custom scripts
from .utils import build_sparse_matrix, Node
from ..utils import load_folder, dump_folder, Folder_Type
from ..constants import ignore_class 
# typing imports
from typing import Dict, List, Tuple


class Label_Builder:
	
	"""
	This class is responsible in parsing the labels of the different nodes
	"""
	def __init__(self, doc2node: Dict[str, Node], doc2id: Dict[str, int]):
		"""
		:param doc2node : map from each document name to its corresponding node features/labels/connections
		:param doc2id   : map from document Id to its corresponding row in any tensor
		"""
		self.encoder = LabelEncoder()
		self.labels = self.__extract_labels(doc2node=doc2node, doc2id=doc2id)
		self.__map_labels()
		self.__donwsample_labels()

	def __extract_labels(self, doc2node: Dict[str, Node], doc2id: Dict[str, int]) -> List[str]:
		"""
		:param doc2node : map from each document name to its corresponding node features/labels/connections
		:param doc2id   : map from document Id to its corresponding row in any tensor
		:returns: a list of labels per document

		This function extracts labels per Node
		"""
		label_vec = [None] * len(doc2node)

		for doc_id, doc_node in doc2node.items():
			
			doc_row_idx = doc2id[doc_id]
			label_vec[doc_row_idx] = doc_node.label

		return label_vec

	def __map_labels(self) -> None:
		"""
		This function maps the labels from strings to integers
		"""
		self.labels = self.encoder.fit_transform(self.labels)

	def __donwsample_labels(self, nsamp_pcls: int=60) -> None:
		"""
		:param nsamp_cls: number of samples per class

		This function as within the GCN paper keeps 20 samples per class
		"""
		unique_cls = np.unique(self.labels)
		for cls_ in unique_cls:
			# find labels of current class
			indices = np.where(self.labels == cls_)[0]
			# randomly select all indices except of those of the k samples
			indices = np.random.choice(indices, size=indices.shape[0] - nsamp_pcls, replace=False)
			# unlabel those samples
			self.labels[indices] = ignore_class

	def get_labels(self) -> torch.Tensor:
		"""
		:returns: a tensor with all labels per node
		"""
		return torch.from_numpy(self.labels).to(torch.long)


class Feat_Builder:
	
	"""
	This class builds the feature matrix out of the node class
	"""
	def __init__(self, doc2node: Dict[str, Node], doc2id: Dict[str, int]):
		"""
		:param doc2node : map from each document name to its corresponding node features/labels/connections
		:param doc2id   : map from document Id to its corresponding row in any tensor
		"""
		self.X = self.__build_matrix(doc2node=doc2node, doc2id=doc2id)

	def __build_matrix(self, doc2node: Dict[str, Node], doc2id: Dict[str, int]):
		"""
		:param doc2node : map from each document name to its corresponding node features/labels/connections
		:param doc2id   : map from document Id to its corresponding row in any tensor

		This function builds up the feature matrix
		"""
		X_rows, X_cols = [], [0]
		for doc_id, doc_node in doc2node.items():

			# extract indices of 1's of one hot feature encoding of tokens
			row_idx_lst = doc_node.feat_one_hot
			# get row index of document
			col_idx = doc2id[doc_id]
			col_idx_lst = X_cols[-1] + len(row_idx_lst)
			# add those to the existing list
			X_rows += row_idx_lst
			X_cols.append(col_idx_lst)

		# note due to the structure of the node it is simpler to build the matrix as N_one_hotxN_docs
		# then transpose it (not most efficient but for this small dataset it is fine)
		mat = build_sparse_matrix(row_indices=X_rows, col_indices=X_cols)
		mat = mat.transpose(0, 1)
		return mat

	def get_feat_matrix(self) -> torch.Tensor:
		return self.X


class Adjacency_Builder:

	"""
	This class builds the adjacency matrix of the graph; needed for message passing
	"""
	def __init__(self, doc2node: Dict[str, Node], doc2id: Dict[str, int]):
		"""
		:param doc2node : map from each document name to its corresponding node features/labels/connections
		:param doc2id   : map from document Id to its corresponding row in any tensor
		"""		
		self.A = self.__build_adjacency(doc2node=doc2node, doc2id=doc2id)


	def __build_adjacency(self, doc2node: Dict[str, Node], doc2id: Dict[str, int]) -> torch.Tensor:
		"""
		:param doc2node : map from each document name to its corresponding node features/labels/connections
		:param doc2id   : map from document Id to its corresponding row in any tensor
		:returns: adjacency matrix which is a sparse matrix
		"""		
		A_rows, A_cols = [], [0]
		for doc_id, doc_node in doc2node.items():

			# extract list of neighbors of current node and map them to their indices in the matrix
			nei_idx_lst = list(map(lambda node_idx: doc2id[node_idx], doc_node.connected_to))
			# get the row id of current node
			col_idx = doc2id[doc_id]
			col_idx_lst = A_cols[-1] + len(nei_idx_lst)
			# add those to the existing list
			A_rows += nei_idx_lst
			A_cols.append(col_idx_lst)

		return build_sparse_matrix(row_indices=A_rows, col_indices=A_cols)

	def get_adjacency(self):
		return self.A


def build_doc2id_mapper(doc2node: Dict[str, Node]) -> Dict[str, int]:
	"""
	:param doc2node: a dicitionary mapping each document Id to the corresponding Node
	:returns: a dictionary mapping from row Id in matrix to the document Id
	"""
	doc2id = {}
	for idx, k in enumerate(doc2node.keys()):
		doc2id[k] = idx
	return doc2id


def generate_inference_masks(label_tens: torch.Tensor, mask_tens:torch.Tensor=None, 
							 num_inf_samp:int=20) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	:param label_tens : tensor having labels per node in graph
	:param mask_tens  : indices of samples to be excluded from consideration
	:param split_size : the size of the inference split
	:returns: the indices of train and inference splits
	"""
	# extract classes in the dataset
	unique_cls = torch.unique(label_tens)
	unique_cls = unique_cls[unique_cls != ignore_class]

	train_lst, test_lst = [], []

	for target_class in unique_cls:

		# keep nodes that have this class label
		marked_nodes_idx = torch.argwhere(label_tens == target_class).squeeze()

		# mask out nodes that need to be excluded
		if mask_tens is not None:
			bool_mask = torch.isin(marked_nodes_idx, mask_tens)
			marked_nodes_idx = marked_nodes_idx[~bool_mask]


		# generate a range from 0 till length of labeled nodes
		num_marked_nodes = marked_nodes_idx.shape[0]
		marked_range = torch.randperm(num_marked_nodes)

		# split into train and test indices
		train_idx, test_idx = marked_range[num_inf_samp:], marked_range[:num_inf_samp]
		# keep the train and validation indices in the main tensor
		train_nodes, test_nodes = marked_nodes_idx[train_idx] , marked_nodes_idx[test_idx]

		train_lst.append(train_nodes)
		test_lst.append(test_nodes)

	# concatenate the indices across all classess
	train_nodes = torch.cat(train_lst, axis=0)
	test_nodes  = torch.cat(test_lst, axis=0)

	# shuffle the tensors
	train_nodes = train_nodes[torch.randperm(train_nodes.shape[0])]
	test_nodes  = test_nodes[torch.randperm(test_nodes.shape[0])]

	return train_nodes, test_nodes


def run_pipeline(data_conf_path: str) -> None:
	"""
	:param data_conf_path: path having configuration of all data related folders

	This function iterates the generation of all data related components => 
	features/adjacency/degree/train-val-test splits
	"""
	cwd = os.getcwd()
	# read path with configurations
	data_conf = load_folder(path_name=data_conf_path, dtype=Folder_Type.yaml_f)

	# load map from document Id to Node class
	doc2node = load_folder(path_name=f"{cwd}/{data_conf['artefact_dir']}/{data_conf['artefact_node_dct']}",\
						   dtype=Folder_Type.pickle_f, is_bytes=True)

	# generate a map to link each document Id to an index number used to represent this node in the graph later
	id2doc = build_doc2id_mapper(doc2node=doc2node)

	# geneate features, adjacency and labels
	gt_builder = Label_Builder(doc2node=doc2node, doc2id=id2doc)
	label_tens = gt_builder.get_labels()

	ft_builder = Feat_Builder(doc2node=doc2node, doc2id=id2doc)
	X_tens = ft_builder.get_feat_matrix()

	am_builder = Adjacency_Builder(doc2node=doc2node, doc2id=id2doc)
	A_tens = am_builder.get_adjacency()

	# generate degree matrix
	D_tens = torch.matmul(A_tens, torch.ones(A_tens.size()[0], dtype=torch.float64))
	#D_tens = torch.sparse.spdiags(D_tens, torch.Tensor([0]).to(torch.long), tuple(A_tens.size())) 
	
	# generate tensors having nodes for train/val/test splits
	train_nodes, test_nodes = generate_inference_masks(label_tens=label_tens)
	train_nodes, val_nodes  = generate_inference_masks(label_tens=label_tens, mask_tens=test_nodes)

	artefacts = {'feats': X_tens.to_dense(), 'adjacency': A_tens.to_dense(), 'degree':D_tens,
				 'label': label_tens, 'train_split': (train_nodes, val_nodes), 'test_split': test_nodes}

	dump_folder(path_name=f"{cwd}/{data_conf['artefact_dir']}/{data_conf['artefact_torch']}",
				obj=artefacts, dtype=Folder_Type.torch_f, is_bytes=True)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--conf_path', type=str, help='Path having data configuration')
	args = parser.parse_args()

	run_pipeline(data_conf_path=args.conf_path)

