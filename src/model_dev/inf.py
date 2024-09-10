import torch
import pandas as pd
# custom scripts imports
from .models import GCN
from ..utils import load_folder, Folder_Type
from ..constants import ignore_class
from ..data.data_splitter import Splitter
# typing imports
from typing import Dict


class Inference_Engine:

	"""
	This class could be used to quantify the quality of a model over an existing test set or even generate
	online predictions for newly introduced nodes in the existing graph
	"""

	def __init__(self, mdl_exp_dir:str):
		"""
		:param mdl_exp_dir: the directory having the model's experiment (with weights and hp configuration)
		"""	
		self.exp_dir = mdl_exp_dir
		self.mdl = self.__load_trained_model()

	def __load_trained_model(self) -> torch.nn.Module:
		"""
		:returns: the model with its training weights loaded
		"""
		hp_exp = load_folder(path_name=f'{self.exp_dir}/hp_configuration.yaml', dtype=Folder_Type.yaml_f)
		mdl = GCN(**hp_exp['GCN'])
		mdl.load_state_dict(load_folder(path_name=f'{self.exp_dir}/mdl_wts.pt', dtype=Folder_Type.torch_f))
		mdl.eval()
		return mdl

	def __predict(self, X_test: torch.Tensor, Graph_tens : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
		"""
		:param X_test     : the test split of the data
		:param Graph_tens : the graph representing tensors (adjacency and degree)
		:returns : the predicted class of each node in the graph
		"""
		A_full, D_full = Graph_tens
		# row normalize features 
		X = F.normalize(X_test, p=2.0, dim=1)
		with torch.no_grad():
			# compute the message passing matrix
			mp_fn = torch.mm(torch.mm(D_full, A_full), D_full)
			# forward the data into the model
			logits = self.mdl(X=X, MP=mp_fn)
			# compute predicted class per node
			preds = torch.argmax(F.softmax(logits.detach(), dim=1), dim=1)
			return preds


	def evaluate(self, dsplit_module: Splitter) -> None:
		"""
		:param dsplit_module: module used across pipeline to generate splits of data

		Generates a classification report having the evaluation of the model on the inference set
		"""
		# get inference split
		(X, A, D), labels = dsplit_module.get_test_split()
		labels = labels.numpy()
		# get preds
		preds = self.__predict(X_test=X, Graph_tens=(A, D))
		preds = preds.numpy()

		# test node indices
		idx = np.where(labels != ignore_class)

		# compute quality of predictions
		cls_rep_df = pd.DataFrame(classification_report(y_true[idx], y_pred[idx], output_dict=True))
		cls_rep_df.to_csv(f'{self.exp_dir}/test_performance.csv')

		