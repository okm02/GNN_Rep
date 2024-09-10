import os, argparse
# custom scripts
from .utils import Node
from ..utils import load_folder, dump_folder, Folder_Type
# typing imports
from typing import List


class Data_Loader:

	"""
	This class is responsible in parsing the raw CITESEER dataset files into Node data structure
	"""

	def __init__(self, data_dir: str, node_fname: str, edge_fname: str, dump_path: str):
		"""
		:param data_dir   : path to directory having the dataset
		:param node_fname : file name having node features
		:param edge_fname : file name having node connections
		:param dump_path  : path to dump artefacts to
		"""
		self.node_path = f'{data_dir}/{node_fname}'
		self.edge_path = f'{data_dir}/{edge_fname}'
		self.doc2node = {}
		self.dump_path = dump_path
		
	def __parse_node_file(self) -> None:
		"""
		This function builds the nodes of the graphs and their corresponding features
		"""
		def parse_one_hot(one_hot_lst: List[str]) -> List[int]:
			"""
			:param one_hot_lst: a one-hot embedding of tokens per document
			:returns : a list of indices that have a labeled token
			"""
			one_hot_lst = list(map(int, one_hot_lst))
			# map each 1 in the hot embedding vector to its index
			one_hot_lst = list(map(lambda x: x[0] if x[1] == 1 else - 1,enumerate(one_hot_lst)))
			one_hot_lst = list(filter(lambda x: x>=0, one_hot_lst))
			return one_hot_lst

		# read folder
		f_contents = load_folder(path_name=self.node_path, dtype=Folder_Type.text_f)

		# parse contents
		for line in f_contents:

			doc_info = line.split()
			# parse each line as in README file
			doc_id, doc_feats, doc_label = doc_info[0], doc_info[1:-1], doc_info[-1]
			# create new node and initialize its contents
			self.doc2node[doc_id] = Node(feat_one_hot=parse_one_hot(one_hot_lst=doc_feats), 
																	label=doc_label,
																	connected_to=[doc_id]) 

	def __parse_edge_file(self) -> None:
		"""
		This function stores per node the list of neighboring nodes it is connected to
		"""
		# read folder
		f_contents = load_folder(path_name=self.edge_path, dtype=Folder_Type.text_f)
		# parse contents
		for line in f_contents:

			# parse contents
			edge_info = tuple(line.split())
			en1, en2 = edge_info

			# add undirected edges from each document to the other
			if (en1 in self.doc2node) and (en2 in self.doc2node):
				self.doc2node[en1].connected_to.append(en2)
				self.doc2node[en2].connected_to.append(en1)

	def generate_data_artefact(self):
		"""
		This function generates the dictionary mapping from document Id to all its features + 
		connections with graph
		"""
		self.__parse_node_file()
		self.__parse_edge_file()

		dump_folder(path_name=self.dump_path, obj=self.doc2node, dtype=Folder_Type.pickle_f, is_bytes=True)


def run_pipeline(path_conf: str) -> None:
	"""
	:param path_conf: path to data configuration folder
	
	This function reads both features and adjacency and dumps the graph to a folder
	"""
	data_conf = load_folder(path_name=path_conf, dtype=Folder_Type.yaml_f)

	if not os.path.exists(data_conf['artefact_dir']):
		os.mkdir(data_conf['artefact_dir'])

	
	loader = Data_Loader(data_dir=data_conf['data_dir'], node_fname=data_conf['node_fname'],\
						 edge_fname=data_conf['edge_fname'], 
						 dump_path=f"{data_conf['artefact_dir']}/{data_conf['artefact_node_dct']}")

	loader.generate_data_artefact()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--conf_path', type=str, help='Path having data configuration')
	args = parser.parse_args()

	run_pipeline(path_conf=args.conf_path)
	
