import scipy.io as sio
import sys
import os
import re
import numpy as np
import torch
from torch.utils import data
from os import listdir
from os.path import isfile, join



### The naming convention of a file is "ai_sj_tk_modality",
# where ai stands for action number i,
# sj stands for subject number j,
# tk stands for trial k,


# For evaluation (Chen et al. 2015),
# we used subjects 1, 3, 5, 7 for training and subjects 2, 4, 6, 8 for testing.



project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_directory)
data_path = (project_directory + "/data/Inertial/")
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]


subject = []
train_set = []
test_set = []

for ex in onlyfiles:

    sub = re.search('_s(.*)_t', ex)
    subject.append(int(sub.group(1)))


    if int(sub.group(1)) in [1,3,5,7]:

        train_set.append(ex)

    else:
        test_set.append(ex)









def create_data_loaders(data_path,
                        list_of_seq,
                        batch_size,
                        take_n_examples = -1,
                        window = 32,
                        ):
    """

    :param data_path:
    :param list_of_seq:
    :param batch_size:
    :param take_n_examples:
    :param window:
    :return: DataLoader
    """

    if take_n_examples > 0:
        list_of_seq = list_of_seq[:take_n_examples]




    # Training set
    set = TorchDataset(data_path,
                        list_of_seq,
                        window = window)


    set_loader = data.DataLoader(set,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=4)

    return set_loader





class TorchDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_path,
                 action_list,
                 window=32):
        """

        :param data_path:
        :param action_list:
        :param window:
        """


        super(TorchDataset, self).__init__()

        self.data_path = data_path
        self.action_list = action_list

        # Length of the desired input for the network
        self.window = window



    # Zero padding

    def __getitem__(self, example_id):
        """

        :param example_id:
        :return: zero-padded examples (X) and labels (Y) [X, Y]
        """

        act = re.search('a(.*)_s', self.action_list[example_id])
        Y = int(act.group(1))
        Y = int(Y-1)

        buff_seq = np.zeros((self.window, 6))
        seq = sio.loadmat(self.data_path+self.action_list[example_id]).get('d_iner')
        buff_seq[0:seq.shape[0], :] = seq
        X = buff_seq



        return X, Y


    def __len__(self):
        return len(self.action_list)


