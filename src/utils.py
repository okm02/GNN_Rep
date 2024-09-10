import torch, yaml
import pickle as pkl
from enum import Enum
# typing imports
from typing import Any


class Folder_Type(Enum):
	pickle_f = 1
	yaml_f   = 2
	torch_f  = 3
	text_f   = 4


def load_folder(path_name: str, dtype:Folder_Type, is_bytes:bool=False) -> Any:
	"""
	:param path_name : path to folder to read
	:param dtype     : type of folder to load (an enumerator that encodes file types used across code)
	:param is_bytes  : if folder contains bytes
	:returns: the contents of the folder
	"""
	try:

		if dtype == Folder_Type.torch_f:
			return torch.load(path_name)
		else :

			# open stream and read data
			with open(path_name, 'rb' if is_bytes else 'r') as f:

				if dtype == Folder_Type.pickle_f:
					return pkl.load(f)
			
				elif dtype == Folder_Type.yaml_f:
					return yaml.safe_load(f)
			
				elif dtype == Folder_Type.text_f:
					return f.readlines()

				else:
					raise NotImplementedError

	except Exception as e:
		raise e


def dump_folder(path_name: str, obj: Any, dtype:Folder_Type, is_bytes:bool=False) -> None:
	"""
	:param path_name : path to folder to dump the object to
	:param obj       : the object we are interested in saving
	:param dtype     : type of folder to write to (an enumerator that encodes file types used across code)
	:param is_bytes  : if folder should write bytes
	"""
	try:

		byte_write = 'wb' if is_bytes else 'w'

		if dtype == Folder_Type.torch_f:
			torch.save(obj, path_name)
		else:

			with open(path_name, byte_write) as f:

				if dtype == Folder_Type.pickle_f:
					pkl.dump(obj, f)
				elif dtype == Folder_Type.yaml_f:
					yaml.dump(obj, f)
				else:
					raise NotImplementedError
				
	except Exception as e:
		raise e
