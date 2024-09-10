import os, argparse, itertools, uuid
# custom scripts
from .models import GCN
from .train import Trainer
from ..data.data_splitter import Splitter
from ..utils import Folder_Type, load_folder, dump_folder
from ..constants import ignore_class
# typting imports
from typing import Dict, Tuple, Generator, List, Any


class HP_Grid_Constructor:

	"""
	This class generates a list of hyperparameter configurations to sweep over;
	Each configuration represents a combination of HP to monitor model performance
	"""

	def __init__(self, grid_conf_path: str):
		"""
		:param grid_conf_path : path having the hyper-parameter grid file (yaml format)
		"""
		self.gpath = grid_conf_path
		assert self.gpath.endswith('yaml')

	def __categorize_hp(self, hp_dct: Dict) -> Tuple[Dict, Dict]:
		"""
		:param hp_dct: the yaml loaded dictionary having all hyper-parameters
		:returns: 
					- a dictionary with hyper-parameters to be searched
					- a dictionary with hyper-parameters that are fixed (not critical)
		"""
		# split the hyperparameters into fixed and grid hyperparameters
		# the structure of the file is nested; each component of the training pipeline has
		# its own hyperparameters and hence you find the nested loop (ex: Model: depth: [hp])
		hp_grid_dct  = {(module, hp_k): hp_v for module, module_hp in hp_dct.items() 
						for hp_k, hp_v in module_hp.items() if isinstance(hp_v, list)}
		hp_const_dct = {(module, hp_k): hp_v for module, module_hp in hp_dct.items()
						for hp_k, hp_v in module_hp.items() if not (module, hp_k) in hp_grid_dct}

		return hp_const_dct, hp_grid_dct

	def __generate_hp_combinations(self, grid_dct: Dict) -> List[Dict]:
		"""
		:param grid_dct: a dictionary with values corresponding to the whole grid
		:returns: combinations of all hyper-parameters
		"""
		# generate combinations
		key_lst, value_lst = [], []

		# extract keys and values
		for hp_k, hp_v in grid_dct.items():
			key_lst.append(hp_k)
			value_lst.append(hp_v)

		# generate the combinations
		hp_combinations = itertools.product(*value_lst)

		# build dictionary from the different combination list
		grid = [{key_lst[idx]: hp_c[idx] for idx in range(0, len(hp_c))} for hp_c in hp_combinations]

		return grid

	def __merge_dcts(self, grid_lst: List[Dict], fixed_hp_dct: Dict) -> List[Dict]:
		"""
		:param grid_lst     : a list of dictionaries where each element is a combination of hyperparameters
		:param fixed_hp_dct : a dictionary having fixed hyper-parameters
		:returns: a list of dictionaries nested to have the format in mdl_config.yaml under configuation folder
		"""
		def restore_nested_hp(flat_dct: Dict[Tuple[str, str], Any]) -> Dict:
			"""
			:param flat_dct: dictionary having the flat version of the hp grid (not nested rather keys are tuples)
			:returns: the dictionary having the original format of HP configuration
			"""
			hp_dct = {}
			for (k_ord1, k_ord2), val in flat_dct.items():

				if k_ord1 not in hp_dct:
					hp_dct[k_ord1] = {k_ord2: val}
				else:
					hp_dct[k_ord1][k_ord2] = val
			return hp_dct

		# merge the dictionaries of each hyperparameter configuration and the constant
		full_grid_lst = [{**dct, **fixed_hp_dct} for dct in grid_lst]
		
		# restore the format of the nested dictionary
		for idx in range(0, len(full_grid_lst)):
			curr_hp = full_grid_lst[idx]
			full_grid_lst[idx] = restore_nested_hp(flat_dct=curr_hp)

		return full_grid_lst


	def generate_grid(self) -> List[Dict]:
		"""
		:returns : the configurations per grid

		Automates the generation of all hyperparameter configurations
		"""
		# load the folder
		grid_dct = load_folder(self.gpath, Folder_Type.yaml_f)
		# split the hpyer-parameter dictionary
		const_hp_dct, grid_search_dct = self.__categorize_hp(hp_dct=grid_dct)
		# generate the hyper-parameter combinations
		hp_grid_lst = self.__generate_hp_combinations(grid_dct=grid_search_dct)
		# merge constant with values to sweep over
		full_grid = self.__merge_dcts(grid_lst=hp_grid_lst, fixed_hp_dct=const_hp_dct)
		
		return full_grid


class Sweeper:

	def __init__(self, hp_builder: HP_Grid_Constructor, data_builder: Splitter, dump_dir: str):
		"""
		:param hp_builder   : module for building hyperparameter grid
		:param data_builder : module for generating data per split
		:param dump_dir     : directory to dump sweep artefacts
		"""
		self.hp_builder = hp_builder
		self.data_mod   = data_builder
		self.dump_dir   = dump_dir

	def __generate_exp_id(self, id_len:int=4) -> str:
		"""
		:param id_len: length of experiment ID
		:returns: the id of current experiment
		"""
		rand_id = str(uuid.uuid4())[:id_len]
		return rand_id

	def __create_dir(self, res_dir: str):
		"""
		:param res_dir : directory to dump results to
		Function to generate directory to dump results to
		"""
		if not os.path.exists(res_dir):
				os.makedirs(res_dir)

	def run(self):
		"""
		Run the hyperparameter sweep
		"""
		
		for idx, hp_conf in enumerate(self.hp_builder.generate_grid()):

			print(f'Started HP Sweep {idx}')

			# build path to dump artefacts to
			exp_id = self.__generate_exp_id()
			exp_dump_path = f"{self.dump_dir}/{exp_id}"

			self.__create_dir(res_dir=exp_dump_path)

			# add id to config dict
			hp_conf['Exp_ID'] = exp_id

			dump_folder(f'{exp_dump_path}/hp_configuration.yaml', hp_conf, Folder_Type.yaml_f)

			# intialize model
			mdl = GCN(**hp_conf['GCN'])

			# fit the model
			trainer = Trainer(mdl=mdl, trainer_conf=hp_conf, wts_dump_path=exp_dump_path)
			trainer.fit(dsplit_module=self.data_mod)

			print(f'Completed HP Sweep {idx}')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--artefact_dir', type=str, help='Path having the artefact with data parsed from raw files')
	parser.add_argument('--sweep_grid_path', type=str, help='Path having hyper-parameter grid')
	parser.add_argument('--dump_path', type=str, help='Path to dump results to (train-val logs/HP config/Model weights)')
	args = parser.parse_args()

	sw_mod = Sweeper(hp_builder=HP_Grid_Constructor(grid_conf_path=args.sweep_grid_path),\
					 data_builder=Splitter(artefact_path=args.artefact_dir, ignore_class=ignore_class),\
					 dump_dir=args.dump_path)
	

	sw_mod.run()