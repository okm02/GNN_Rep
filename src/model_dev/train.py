import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# custom scripts
from .logger import CSV_Logger
from ..data.data_splitter import Splitter
from ..constants import ignore_class
from ..utils import dump_folder, Folder_Type
# typing imports
from typing import Tuple, Dict


def compute_accuracy(logits: torch.Tensor, gt_tens: torch.Tensor) -> torch.Tensor:
	"""
	:param logits  : model logits 
	:param gt_tens : ground truth vector
	:returns : the accuracy of the model
	"""
	# compute probabilities and keep class with highest probability
	preds = torch.argmax(F.softmax(logits.detach(), dim=1), dim=1)
	# find indices of actual classes
	gt_indx = torch.nonzero(gt_tens != ignore_class).squeeze()
	# compute accuracy
	acc = (preds[gt_indx] == gt_tens[gt_indx])
	acc = acc.sum()/acc.shape[0]

	return acc


class Trainer:

	"""
	This class encapsulates all training/inference logic
	"""

	def __init__(self, mdl: torch.nn.Module, trainer_conf: Dict, wts_dump_path: str):
		"""
		:param mdl           : the model to train
		:param trainer_conf  : the number of epochs to train the model on
		:param wts_dump_path : the directory to dump weights to
		"""
		self.mdl = mdl
		self.optimizer = torch.optim.Adam(self.mdl.parameters(), **trainer_conf['OPT'])
		self.num_epochs = trainer_conf['Pipe']['epochs']
		self.logger = SummaryWriter(log_dir=wts_dump_path)
		#CSV_Logger(dump_dir=wts_dump_path)
		self.dump_dir = wts_dump_path

	def __mask_nodes(self, feats: torch.Tensor, graph: Tuple, mask: Tuple):
		"""
		:param feats : the node feature matrix
		:param graph : a tuple having the adjacency and the degree matrices
		:param mask  : nodes that need to be masked while training
		:returns : the features/adjacency/degree matrices with nodes not belonging to split masked
		"""
		(A, D) = graph
		Xm, Am, Dm = mask

		# mask the nodes which model should not be exposed to
		split_X = torch.mul(feats, Xm) 
		split_A = torch.mul(A, Am)
		split_D = torch.mul(D, Dm)

		return split_X, split_A, split_D

	def __compute_metrics(self, feats: torch.Tensor, adj: torch.Tensor, deg: torch.Tensor, label_tens: torch.Tensor):
		"""
		:param feats      : the node features matrix
		:param adj        : the adjacnecny matrix of the graph
		:param deg        : the degree matrix of the graph
		:param label_tens : tensor having the ground truth
		:returns: the loss and accuracy of the model on current batch
		"""
		# compute the message passing matrix
		mp_fn = torch.mm(torch.mm(deg, adj), deg)
		# forward the data into the model
		logits = self.mdl(X=feats, MP=mp_fn)

		# compute loss of the model
		loss = F.cross_entropy(logits, label_tens, ignore_index=ignore_class)

		# compute accuracy
		acc = compute_accuracy(logits=logits, gt_tens=label_tens)
		
		return loss, acc

	def __train_step(self, data: Tuple, graph: Tuple, train_masks: Tuple, epoch:int) -> None:
		"""
		:param data        : a tuple having the train node features and labels
		:param graph       : a tuple having the adjacency and the degree matrices
		:param train_masks : nodes that need to be masked while training
		:param epoch       : the current epoch to log the metrics
		"""
		(X, y_train) = data
		(A, D) = graph

		# mask non-train nodes
		train_X, train_A, train_D = self.__mask_nodes(feats=X, graph=graph, mask=train_masks)

		loss, acc = self.__compute_metrics(feats=train_X, adj=train_A, deg=train_D, label_tens=y_train)

		self.logger.add_scalar('Loss/train', loss.detach().item(), epoch)
		self.logger.add_scalar('Acc/train', acc.item(), epoch)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def __val_step(self, data: Tuple, graph: Tuple, val_masks: Tuple, epoch:int) -> None:
		"""
		:param data        : a tuple having the validation node features and labels
		:param graph       : a tuple having the adjacency and the degree matrices
		:param val_masks   : nodes that need to be masked while validation
		:param epoch       : current epoch to log metrics
		"""
		(X, y_val) = data
		(A, D) = graph

		# mask the nodes which model should not be exposed to
		val_X, val_A, val_D = self.__mask_nodes(feats=X, graph=graph, mask=val_masks)

		with torch.no_grad():

			loss, acc = self.__compute_metrics(feats=val_X, adj=val_A, deg=val_D, label_tens=y_val)

			self.logger.add_scalar('Loss/val', loss.detach().item(), epoch)
			self.logger.add_scalar('Acc/val', acc.item(), epoch)

	def __dump_weights(self):
		"""
		This function saves the model weights for later use
		"""
		dump_folder(path_name=f'{self.dump_dir}/mdl_wts.pt', obj=self.mdl.state_dict(), dtype=Folder_Type.torch_f)


	def fit(self, dsplit_module: Splitter):
		"""
		:param dsplit_module : module for providing us with train/val/inference splits

		Runs the full training loop
		"""

		# get data needed to train the model
		(X, A, D), (y_train, y_val), train_masks, val_masks = dsplit_module.get_train_split()

		# row normalize features 
		X = F.normalize(X, p=2.0, dim=1)

		for e in range(0, self.num_epochs):

			# switch to train
			self.mdl.train()
			self.__train_step(data=(X, y_train), graph=(A, D), train_masks=train_masks, epoch=e)
			self.mdl.eval()
			self.__val_step(data=(X, y_val), graph=(A, D), val_masks=val_masks, epoch=e)

		self.logger.close()
		self.__dump_weights()

	
	def predict(self, dsplit_module: Splitter):

		(X, A, D), y_test = dsplit_module.get_test_split()

		# row normalize features 
		X = F.normalize(X, p=2.0, dim=1)

		# compute test loss and accuracy
		loss, acc = self.__compute_metrics(feats=X, adj=A, deg=D, label_tens=y_test)

		print(f'Test Loss = {loss} ; Test accuracy = {acc}')