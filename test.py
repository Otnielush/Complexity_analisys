import sys
sys.path.insert(0, './sequence-labeler-master')

from complex_labeller import Complexity_labeller
model_path = './cwi_seq.model'
temp_path = './temp_file.txt'

import numpy as np