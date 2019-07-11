
from utility.training import training_loop
import yaml

f = open('sample_configs/model_f.yaml', 'r')
config=yaml.load(f)
training_loop(**config)

