# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
import numpy as np
args = Namespace()

def log_func(x, a, b):
  return a + b * np.log(x)

arg_dict = {
    'run_gradio': False, 
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    #'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 100, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.5, 
    'delta': 2,
    'alpha': 1, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 3.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'soft_colour': True,
}

args.__dict__.update(arg_dict)

from demo_watermark import main

#main(args)


x_values1 = []
y_values1 = []
args.__dict__.update(arg_dict)
x_values1, y_values1 = (main(args))


arg_dict = {
    'run_gradio': False, 
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    #'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 100, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.5, 
    'delta': 2.5,
    'alpha': 2, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 3.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'soft_colour': True,
}
x_values2 = []
y_values2 = []
args.__dict__.update(arg_dict)
x_values2, y_values2 = (main(args))


arg_dict = {
    'run_gradio': False, 
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    #'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 100, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.5, 
    'delta': 3,
    'alpha': 2, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 3.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'soft_colour': True,
}
x_values3 = []
y_values3 = []
args.__dict__.update(arg_dict)
x_values3, y_values3 = (main(args))

arg_dict = {
    'run_gradio': False, 
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    #'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 100, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.5, 
    'delta': 5,
    'alpha': 3, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 3.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'soft_colour': True,
}
x_values4 = []
y_values4 = []
args.__dict__.update(arg_dict)
x_values4, y_values4 = (main(args))




arg_dict = {
    'run_gradio': False, 
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    #'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 100, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.5, 
    'delta': 9,
    'alpha': 7, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 3.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'soft_colour': True,
}
x_values5 = []
y_values5 = []
args.__dict__.update(arg_dict)
x_values5, y_values5 = (main(args))

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
auc = np.trapz(y_values1, x_values1)
temp = "greenlist bias = 2, redlist bias = 1, greenlist size = 0.5, AUC =" + str(auc)
plt.plot(x_values1, y_values1, color='red', marker='o', linestyle='-', label=temp)

auc = np.trapz(y_values2, x_values2)
temp = "greenlist bias = 2.5, redlist bias = 2, greenlist size = 0.5, AUC =" + str(auc)
plt.plot(x_values2, y_values2, color='green', marker='o', linestyle='-',label=temp)

auc = np.trapz(y_values3, x_values3)
temp = "greenlist bias = 3, redlist bias = 2, greenlist size = 0.5, AUC =" + str(auc)
plt.plot(x_values3, y_values3, color='blue', marker='o', linestyle='-',label=temp)

auc = np.trapz(y_values4, x_values4)
temp = "greenlist bias = 5, redlist bias = 2, greenlist size = 0.5, AUC =" + str(auc)
plt.plot(x_values4, y_values4, color='yellow', marker='o', linestyle='-',label=temp)

auc = np.trapz(y_values5, x_values5)
temp = "greenlist bias = 9, redlist bias = 7, greenlist size = 0.5, AUC =" + str(auc)
plt.plot(x_values5, y_values5, color='purple', marker='o', linestyle='-',label=temp)
# Plot the best-fitting curve
plt.xlabel('False Positve Rate')
plt.ylabel('True Negative Rate')
plt.grid(True)
plt.legend()
plt.show()
