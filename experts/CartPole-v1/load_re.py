import torch
import os
from pprint import  pprint
#from agents.safe_sarsa_agent import *
dir_path = os.path.dirname(__file__)

print(dir_path)
path1 = dir_path+ '/policy.ckpt'

lz = torch.load(dir_path + '/policy.ckpt')

pprint(lz)
print(lz['OrderedDict'])
#lz1 = torch.load(dir_path  + '/results_dict.pt')

#pprint(lz1)

