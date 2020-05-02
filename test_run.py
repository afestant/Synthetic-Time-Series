import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
from ts_dataset import TSDataset
from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator
from tsgan import TSGANSynthetiser
import yaml

#load config file
path_to_yaml = 'tsgan_configuration.yml'
try: 
    with open (path_to_yaml, 'r') as file:
        config = yaml.safe_load(file)
except Exception:
    print('Error reading the config file')

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{config['run_tag']}_{date}" if config['run_tag'] != '' else date
log_dir_name = os.path.join(config['logdir'], run_name)
writer = SummaryWriter(log_dir_name)

try:
    os.makedirs(config['outf'])
except OSError:
    pass
try:
    os.makedirs(config['imf'])
except OSError:
    pass

data_dir = config['dataset']['path']
filename = config['dataset']['filename']
path_file = data_dir + filename
datetime_col = config['dataset']['datetime_col']
value_col = config['dataset']['value_col']
time_window = config['dataset']['time_window']

tsgan = TSGANSynthetiser(path_to_yaml, writer)
tsgan.fit(path_file, datetime_col, value_col, time_window)
n = config['size']
sample = tsgan.sample_data(n)