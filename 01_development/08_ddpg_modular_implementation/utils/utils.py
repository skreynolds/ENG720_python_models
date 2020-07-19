import yaml


def read_config(path):
	"""
	Return python dict from .yml file

	Input:
	======
	- path (str): path to the .yml config

	Output:
	=======
	- cfg (dict): configuration object
	"""
	with open(path, 'r') as ymlfile:
		cfg = yaml.load(ymlfile)
	return cfg