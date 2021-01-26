import torch
import torch.nn as nn



def weights_init(m):
    """
    Standard module's weight initialization
    :param m: pytorch module
    """

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, gain=1)


class Model(nn.Module):
    """

    Architecture of the model. It includes the 5 convolution blocks, the parsing strategy in order to learn a specific
    set of parameters for each action through grouped convolutions and both binary and multiclass classifiers

    """

    def __init__(self, dummy_input,
                 f_1,
                 f_2,
                 f_3,
                 f_4,
                 f_5,
                 nb_class,
                 bin_layer,
                 multi_layer

):
        """
        Initialization of the model

        :param dummy_input: Dummy input used to automatically initialize the model
        :param f_1: The number of filters of the first convolution block
        :param f_2: The number of filters of the second convolution block
        :param f_3: The number of filters of the third convolution block
        :param f_4: The number of filters of the forth convolution block
        :param f_5: The number of filters of the fifth convolution block
        :param nb_class: The number of class of the data set
        :param bin_layer: The number of neurons in the fully connected layers of the binary classifier
        :param multi_layer: The number of neuros in the fully connected layers of the multiclass classifier
        """
        super().__init__()
        self.__name__ = 'hierarchical CNN'


        ################################################## Input formating #############################################
        print(f'The size of the input is : {dummy_input.size()}')
        x = dummy_input

        # Permute the input in order to adopt the following order :
        # Batch_examples / Signals (acceleration and angular variation rate) / time frames

        x = x.permute(0,2,1)
        print(f'After permutation : {x.size()}')

        ################################################### Block and groups 1 #########################################

        # First convolution block
        class conv_block_1(nn.Module):
            def __init__(self, ):
                super(conv_block_1, self).__init__()

                self.block_1 = nn.Sequential(
                    nn.Conv1d(x.size()[1], f_1, kernel_size=1,  bias=False),
                    nn.SELU(),
                    nn.BatchNorm1d(f_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.MaxPool1d(2))

            def forward(self, x):
                """

                :param x: The previous time series
                :return: The processed time series by the convolution block's operations
                """
                out = self.block_1(x)
                return out

        # First block convolution group (one block for each action)
        class block_1_groups(nn.Module):
            def __init__(self, nb_group):
                super(block_1_groups, self).__init__()
                # Initialize the first block for each group
                self.group = nn.ModuleList(
                    [conv_block_1() for _ in range(nb_group)])

            def forward(self, x):
                """

                :param x: The concatenated previous time series of all the groups
                :return: The processed time series for each group and the sampled features
                """

                for i, pose in enumerate(self.group):

                    if i < 1:
                        out = pose(x)
                    else:
                        out = torch.cat((out, pose(x)), dim=1)

                max_block_1 = torch.max(out,2)
                min_block_1 = torch.min(out, 2)

                return [out, max_block_1, min_block_1]

        # Initialize the first convolution block for all the groups
        self.block_1_groups = block_1_groups(nb_class)

        # Apply the first convolution block operations
        group_conv_1 = self.block_1_groups(x)

        # The outputted processed time series which is fed to the next block
        x = group_conv_1[0]

        # The feature vector's sampled elements of the first time series for each group
        max_block_1 = group_conv_1[1]
        min_block_1 = group_conv_1[2]


        print(f'The output of the first group is of size: {x.size()}')

    ################################################### Block and groups 2 #############################################

        # Second convolution block
        class conv_block_2(nn.Module):
            def __init__(self, ):
                super(conv_block_2, self).__init__()

                self.create_pose_mixer_conv = nn.Sequential(
                    nn.Conv1d(f_1, f_2, kernel_size=3, bias=False),
                    nn.BatchNorm1d(f_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.MaxPool1d(2))

            def forward(self, x):
                out = self.create_pose_mixer_conv(x)
                return out


        # Grouped convolution of the second block
        class block_2_groups(nn.Module):
            def __init__(self, nb_group, f_1):
                super(block_2_groups, self).__init__()

                self.input_group_size = f_1

                self.group = nn.ModuleList(
                    [conv_block_2() for _ in range(nb_group)])


            def forward(self, x):

                for i, pose_mixer in enumerate(self.group):

                    if i < 1:
                        out = pose_mixer(x[:,0:self.input_group_size,:])

                    else:
                        out = torch.cat((out, pose_mixer(x[:,i*self.input_group_size:(i+1)*self.input_group_size,:])),
                                        dim=1)



                # Gives the maximum activation value for each channels in the time series
                max_block_2 = torch.max(out, 2)
                min_block_2 = torch.min(out, 2)
                return [out, max_block_2, min_block_2]


        # Initialize the second convolution block for all the groups
        self.block_2_groups = block_2_groups(nb_class, f_1)

        # Apply the first convolution block operations
        group_conv_2 = self.block_2_groups(x)

        # The outputted processed time series which is fed to the next block
        x = group_conv_2[0]

        # The feature vector's sampled elements of the second time series for each group
        max_block_2 = group_conv_2[1]
        min_block_2 = group_conv_2[2]

        print(f'The output of the second group is of size: {x.size()}')

        ################################################### Block and groups 3  ########################################

        # Third convolution block
        class conv_block_3(nn.Module):
            def __init__(self, ):
                super(conv_block_3, self).__init__()
                self.create_motionlet_conv = nn.Sequential(
                    nn.Conv1d(f_2, f_3, kernel_size=5, bias=False),
                    nn.BatchNorm1d(f_3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.MaxPool1d(2))
            def forward(self, x):
                out = self.create_motionlet_conv(x)
                return out

        # Grouped convolution of the third block
        class block_3_groups(nn.Module):
            def __init__(self,  nb_group, f_3):
                super(block_3_groups, self).__init__()

                self.nb_group = nb_group
                self.input_group_size = f_3
                self.group = nn.ModuleList([conv_block_3() for _ in range(nb_group)])

            def forward(self, x):

                for i, motionlet in enumerate(self.group):
                    if i < 1:
                        out = motionlet(x[:,0:self.input_group_size,:])
                    else:
                        out = torch.cat((out, motionlet(x[:,i*self.input_group_size:(i+1)*self.input_group_size,:])),
                                        dim=1)

                # Gives the maximum activation value for each channels in the time serie
                max_block_3 = torch.max(out, 2)
                min_block_3 = torch.min(out, 2)
                return [out, max_block_3, min_block_3]


        self.block_3_groups = block_3_groups(nb_class, f_3)

        group_conv_3 = self.block_3_groups(x)

        x = group_conv_3[0]

        max_block_3 = group_conv_3[1]
        min_block_3 = group_conv_3[2]

        print(f'The output of the third group is of size: {x.size()}')



        ################################################### Block and groups 4 #########################################

        # Forth convolution block
        class conv_block_4(nn.Module):
            def __init__(self, ):
                super(conv_block_4, self).__init__()

                self.create_actionlet_conv = nn.Sequential(

                    nn.Conv1d(f_3, f_4, kernel_size=7, bias=False),
                    nn.BatchNorm1d(f_4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.MaxPool1d(2))

            def forward(self, x):
                out = self.create_actionlet_conv(x)

                return out


        # Grouped convolution of the forth block
        class block_4_groups(nn.Module):
            def __init__(self, nb_group, input_group_size):
                super(block_4_groups, self).__init__()

                self.input_group_size = input_group_size

                self.nb_group = nb_group

                self.group = nn.ModuleList(
                    [conv_block_4() for _ in range(nb_group)])

            def forward(self, x):

                for i, actionlet in enumerate(self.group):

                    if i < 1:
                        out = actionlet(x[:,0:self.input_group_size,:])

                    else:
                        out = torch.cat((out, actionlet(x[:,i*self.input_group_size:(i+1)*self.input_group_size,:])),
                                        dim=1)


                # Gives the maximum activation value for each channels in the time series
                max_block_4 = torch.max(out,2)
                min_block_4 = torch.min(out,2)

                return [out, max_block_4, min_block_4]

        self.block_4_groups = block_4_groups(nb_class, f_4)


        group_conv_4 = self.block_4_groups(x)

        x = group_conv_4[0]

        max_block_4 = group_conv_4[1]
        min_block_4 = group_conv_4[2]

        print(f'The output of the forth group is of size: {x.size()}')



        ################################################### Block and groups 5 #########################################

        # Fifth convolution block
        class conv_block_5(nn.Module):
            def __init__(self, ):
                super(conv_block_5, self).__init__()

                self.create_card_conv = nn.Sequential(
                    nn.Conv1d(f_4, f_5, kernel_size=4, padding=1, bias=False),
                    nn.BatchNorm1d(f_5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.MaxPool1d(2))

            def forward(self, x):
                out = self.create_card_conv(x)
                return out

        # Grouped convolution of the fifth block
        class block_5_groups(nn.Module):
            def __init__(self, nb_group, input_group_size):

                super(block_5_groups, self).__init__()
                self.input_group_size = input_group_size

                self.nb_group = nb_group

                self.group = nn.ModuleList(
                    [conv_block_5() for _ in range(nb_group)])

            def forward(self, x):

                for i, action_card in enumerate(self.group):

                    if i < 1:
                        out = action_card(x[:,0:self.input_group_size,:])

                    else:
                        out = torch.cat((out, action_card(x[:,i*self.input_group_size:(i+1)*self.input_group_size,:])),
                                        dim=1)

                # Gives the maximum activation value for each channels in the time series
                max_block_5 = torch.max(out,2)
                min_block_5 = torch.min(out,2)

                return [out, max_block_5, min_block_5]


        self.block_5_groups = block_5_groups(nb_class, f_5)

        group_conv_5 = self.block_5_groups(x)

        x = group_conv_5[0]

        max_block_5 = group_conv_5[1]
        min_block_5 = group_conv_5[2]

        print(f'The output of the fifth group is of size: {x.size()}')


        ########################################### Binary max group ###################################################
        class binary_groups(nn.Module):
            def __init__(self, nb_group, f_1, f_2, f_3, f_4, f_5):
                """

                Parse the sampled elements for each group in order to create the feature vector of each binary classifier

                The element of each feature vector are concatenated together and each feature vector is concatenated
                one after the other.

                :param nb_group:
                :param f_1: Number of dimension of the first time series' time step vector
                :param f_2: Number of dimension of the second time series' time step vector
                :param f_3: Number of dimension of the third time series' time step vector
                :param f_4: Number of dimension of the forth time series' time step vector
                :param f_5: Number of dimension of the fifth time series' time step vector
                """
                super(binary_groups, self).__init__()

                self.nb_group = nb_group
                self.f_1 = f_1
                self.f_2 = f_2
                self.f_3 = f_3
                self.f_4 = f_4
                self.f_5 = f_5

            def forward(self, max_block_1, max_block_2, max_block_3, max_block_4, max_block_5,
                                min_block_1, min_block_2, min_block_3, min_block_4, min_block_5):


                for i in range(self.nb_group):


                    if i < 1:

                        out = torch.cat((max_block_1[0][:,0:self.f_1],
                                         max_block_2[0][:, 0:self.f_2],
                                         max_block_3[0][:,0:self.f_3],
                                         max_block_4[0][:,0:self.f_4],
                                         max_block_5[0][:, 0:self.f_5],

                                         min_block_1[0][:, 0:self.f_1],
                                         min_block_2[0][:, 0:self.f_2],
                                         min_block_3[0][:, 0:self.f_3],
                                         min_block_4[0][:, 0:self.f_4],
                                         min_block_5[0][:, 0:self.f_5]),
                                        dim=1)

                    else:

                        out_next = torch.cat((

                            max_block_1[0][:, i * self.f_1:(i + 1) * self.f_1],
                            max_block_2[0][:,i*self.f_2:(i+1)*self.f_2],
                            max_block_3[0][:,i*self.f_3:(i+1)*self.f_3],
                            max_block_4[0][:,i*self.f_4:(i+1)*self.f_4],
                            max_block_5[0][:,i*self.f_5:(i+1)*self.f_5],

                            min_block_1[0][:, i * self.f_1:(i + 1) * self.f_1],
                            min_block_2[0][:, i * self.f_2:(i + 1) * self.f_2],
                            min_block_3[0][:,i * self.f_3:(i + 1) * self.f_3],
                            min_block_4[0][:,i * self.f_4:(i + 1) * self.f_4],
                            min_block_5[0][:,i * self.f_5:(i + 1) * self.f_5]),
                            dim=1)

                        out = torch.cat((out, out_next), dim=1)

                return out


        self.binary_groups = binary_groups(nb_class, f_1, f_2, f_3, f_4, f_5)

        embedding_features = self.binary_groups(max_block_1, max_block_2, max_block_3, max_block_4, max_block_5,
                                                   min_block_1, min_block_2, min_block_3, min_block_4, min_block_5)

        print(f'The embedding_features output size is : {embedding_features.size()}')

        ################################################### Binary classifier ##########################################

        # Size of the feature vector of one convolution group
        feature_size = 2*(f_1 + f_2 + f_3 + f_4 + f_5)
        print(f'The binary features are of size: {feature_size}')

        # The binary classifiers architecture
        class bin_classifier_unit(nn.Module):
            def __init__(self, feature_size):
                super(bin_classifier_unit, self).__init__()
                self.bin_classifier = nn.Sequential(
                    nn.BatchNorm1d(feature_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Linear(feature_size, bin_layer), nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(bin_layer, bin_layer), nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(bin_layer, 2))

            def forward(self, x):
                out = self.bin_classifier(x)
                return out

        # Parse and issue a prediction for each convolution group
        class bin_classifier_group(nn.Module):
            def __init__(self, nb_classes, feature_size):
                super(bin_classifier_group, self).__init__()

                self.group = nn.ModuleList([bin_classifier_unit(feature_size) for _ in range(nb_classes)])
                self.feature_size = feature_size

            def forward(self, x):

                for i, bin_classif in enumerate(self.group):
                    if i < 1:
                        out = self.group[i](x[:, i * self.feature_size:(i + 1) * self.feature_size])
                    else:
                        out_next = self.group[i](x[:, i * self.feature_size:(i + 1) * self.feature_size])
                        out = torch.cat((out, out_next), dim=1)
                return out

        self.bin_classifier_group = bin_classifier_group(nb_class, feature_size)

        x = self.bin_classifier_group(embedding_features)

        ################################################### Multiclass classifier ######################################

        self.multi_classifier = nn.Sequential(
            nn.BatchNorm1d(feature_size*nb_class, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(embedding_features.size()[1],multi_layer ), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(multi_layer, multi_layer), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(multi_layer, nb_class))

        self.apply(weights_init)



    # Forward pass of the hierarchical CNN
    def forward(self, x_seq):


        x = x_seq.permute(0,2,1)

        # Convolution that creates the different time series
        conv_block_1 = self.block_1_groups(x)
        conv_block_2 = self.block_2_groups(conv_block_1[0])
        conv_block_3 = self.block_3_groups(conv_block_2[0])
        conv_block_4 = self.block_4_groups(conv_block_3[0])
        conv_block_5 = self.block_5_groups(conv_block_4[0])

        # Sampled elements of each time series
        max_block_1 = conv_block_1[1]
        min_block_1 = conv_block_1[2]
        max_block_2 = conv_block_2[1]
        min_block_2 = conv_block_2[2]
        max_block_3 = conv_block_3[1]
        min_block_3 = conv_block_3[2]
        max_block_4 = conv_block_4[1]
        min_block_4 = conv_block_4[2]
        max_block_5 = conv_block_5[1]
        min_block_5 = conv_block_5[2]

        # Create the feature vectors for each group
        embedding_features = self.binary_groups(max_block_1, max_block_2, max_block_3, max_block_4, max_block_5,
                                                   min_block_1, min_block_2, min_block_3, min_block_4, min_block_5)

        # Outputs a binary prediction for each group
        out_bin = self.bin_classifier_group(embedding_features)

        # Outputs the multi-class probabilistic prediction vector
        out = self.multi_classifier(embedding_features.detach())



        return [out, out_bin]

