import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# typing imports
from typing import Dict


class GCN(torch.nn.Module):


	def __init__(self, in_dim: int, hid_dim: int, out_dim: int, depth:int, drop_rate:float=0.2):
		"""
		:param in_dim    : input dimension of feature matrix
		:param hid_dim   : hidden dimension of features in deeper layers
		:param out_dim   : the number of classes we wish to predict
		:param depth     : depth of network
		:param drop_rate : dropout strength to apply to feature representations
		"""
		super().__init__()
		self.depth = depth
		self.w_gnn = torch.nn.ParameterDict(self.__build_gnn(d_in=in_dim, d_out=hid_dim, depth=depth))
		self.w_cls = Parameter(data=self.__build_linear_layer(d_in=hid_dim, d_out=out_dim), requires_grad=True)
		self.drop_rate = drop_rate

	def __build_linear_layer(self, d_in: int, d_out: int) -> torch.Tensor:
		"""
		:param d_in  : input dimension of feature matrix
		:param d_out : hidden dimension of features in deeper layers
		:returns: a linear layer

		The need for this custom layer is the ease to compute the product with our current way GCN is formulated
		"""
		linear = torch.empty(d_in, d_out, dtype=torch.float64)
		torch.nn.init.kaiming_uniform_(linear, a=0, mode='fan_in', nonlinearity='relu')
		return linear


	def __build_gnn(self, d_in: int, d_out: int, depth: int) -> Dict[str, torch.nn.Parameter]:
		"""
		:param d_in  : input dimension of feature matrix
		:param d_out : hidden dimension of features in deeper layers
		:param depth : depth of GCN network
		:returns: parameter dictionary per layer
		"""
		param_dct = {}
		in_feat, hid_feat = d_in, d_out
		# this network builds a gnn where first layer maps from dimension X_in to Hid_Dim
		# then all layers have identical input-output dimension H_dim
		for layer_idx in range(0, depth):
			param_dct[f'w_{layer_idx}'] = Parameter(data=self.__build_linear_layer(d_in=in_feat, d_out=hid_feat), requires_grad=True)
			in_feat = hid_feat

		return param_dct

	def forward(self, X: torch.Tensor, MP: torch.Tensor):
		"""
		:param X    : feature matrix of nodes
		:param MP   : matrix to exchange messages across the different nodes according to graph structure
		:returns : computes representations of each node
		"""

		def compute_rep(H_l: torch.Tensor, W:torch.Tensor):
			"""
			:param H_l : the node features at layer L
			:param W   : the weight matrix
			:returns   : the node representations at layer L+1
			"""
			Message  = torch.mm(MP, H_l)
			Node_Rep = torch.mm(Message, W)
			return Node_Rep

		H_l = F.dropout(X, self.drop_rate)
		for layer_idx in range(0, self.depth):
			H_l = F.relu(compute_rep(H_l=H_l, W=self.w_gnn[f'w_{layer_idx}']))
			H_l = F.dropout(H_l, self.drop_rate)

		H_cls = compute_rep(H_l=H_l, W=self.w_cls)
		return H_cls

