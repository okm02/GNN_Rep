import torch
# custom scripts imports
from ..utils import load_folder, Folder_Type
# typing imports
from typing import Dict, Tuple, List


class Splitter:

	"""
	This class is reponsible in splitting the data into the three core splits : train/val/test
	"""
	def __init__(self, artefact_path: str, ignore_class: int):
		"""
		:param artefact_path : path to artefact storing all data checkpoints
		:param ignore_class  : class to ignore when computing the loss
		"""
		self.ignore_class = ignore_class
		# load the artefact dicitionary

		artefact_dct = load_folder(path_name=artefact_path, dtype=Folder_Type.torch_f)

		
		artefact_dct = load_folder(path_name=artefact_path, dtype=Folder_Type.torch_f)
		# load the data
		(self.X, self.Y), (self.A, self.D) = self.__load_data(artefact_dct=artefact_dct)
		# load the splits
		self.tr_split, self.vl_split, self.te_split = self.__load_splits(artefact_dct=artefact_dct)
		# remove the artefact dictionary
		del artefact_dct
		

	def __load_data(self, artefact_dct:Dict) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
		"""
		:param artefact_dct: dictionary with generated artefacts
		:returns : a tuple having node features/labels and tuple having graph structure
		"""
		# load the data components
		Y = artefact_dct['label']
		X = artefact_dct['feats']#.to_sparse_csc()
		A = artefact_dct['adjacency']#.to_sparse_csc()
		# load diagonal values
		D = artefact_dct['degree']
		# compute d^-1/2
		D = torch.reciprocal(torch.sqrt(D))
		# build a diagonal tensor
		D = torch.diag(D)
		#torch.sparse.spdiags(D, torch.Tensor([0]).to(torch.long), tuple(A.size()))

		return (X, Y), (A, D)

	def __load_splits(self, artefact_dct: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		:param artefact_dct : dictionary with generated artefacts
		:returns: tuple having nodes per split
		"""
		(train_nodes, val_nodes), test_nodes = artefact_dct['train_split'], artefact_dct['test_split']
		return train_nodes, val_nodes, test_nodes

	def __mask_label_vector(self, ns_split_lst: List[torch.Tensor]) -> torch.Tensor:
		"""
		:param ns_split_lst : a list of tensors having indices of nodes of other splits we wish to exclude
		:returns : This function replaces node indices in the list with ignore class to exclude their score of the loss
		"""
		label_copy_tens = self.Y.clone().detach()
		for ns_idx_tens in  ns_split_lst:
			# replace tensor indices with mask class
			label_copy_tens[ns_idx_tens] = self.ignore_class
		return label_copy_tens

	def __build_node_mask(self, ns_split_lst: List[torch.Tensor]) -> torch.Tensor:
		"""
		:param ns_split_lst : a list of tensors having indices of nodes of other splits we wish to exclude
		:returns : masks per data split; to cover nodes that should be excluded from this stage
		"""
		mask_feat = torch.ones(self.X.size(), dtype=torch.float)
		mask_adj  = torch.ones(self.A.size(), dtype=torch.float)
		mask_deg  = torch.ones(self.A.size(), dtype=torch.float)

		for indx_tens in ns_split_lst:

			# float('-inf')
			mask_feat[indx_tens, :] = 0.0
			mask_adj[indx_tens, :]  = 0.0
			mask_adj[:, indx_tens]  = 0.0
			mask_deg[indx_tens, indx_tens] = 0.0

		return mask_feat, mask_adj, mask_deg

	def get_train_split(self):
		"""
		Build the train related data splits
		"""
		
		# generate label splits
		train_Y = self.__mask_label_vector(ns_split_lst=[self.vl_split, self.te_split])
		val_Y   = self.__mask_label_vector(ns_split_lst=[self.tr_split, self.te_split])

		# generate masks
		train_masks = self.__build_node_mask(ns_split_lst=[self.vl_split, self.te_split])
		val_masks = self.__build_node_mask(ns_split_lst=[self.te_split])

		return (self.X, self.A, self.D), (train_Y, val_Y), train_masks, val_masks


	def get_test_split(self):
		"""
		Build the test related data splits
		"""
		test_Y = self.__mask_label_vector(ns_split_lst=[self.tr_split, self.vl_split])
		return (self.X, self.A, self.D), test_Y
