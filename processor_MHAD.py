from tqdm import tqdm
# from feeder.dataloader import *
from feeder.utd_mhad_loader import *
from processor.utils.argument_parser import ArgumentParser
from processor.utils.file_io import FileIO
from processor.utils.logger import Logger
from processor.utils.utils import DictAction
import os
import sys
import torch
from os import listdir
from os.path import isfile, join
import re
from thop import profile



project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_directory)

class Processor(ArgumentParser):
    """
    Inspired by https://github.com/yysijie/st-gcn
    Processor that handles
        * Training of the model
        * Testing / evaluation of the model
        * Initializations of the whole training procedure
        * Logging
        * Load & save results / models
    """

    def __init__(self, argv):
        """
        Load the configuration from command line and a specified config file
        Initialize logging, file i/o, model, training environment
        :param argv: arguments from command line
        """
        super().__init__(argv)

        self.logger = Logger(self.work_dir, self.arg.save_log, self.arg.print_log)
        self.fileio = FileIO(self.logger, self.work_dir)
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.eval_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

        self.categorie_names_list = self.load_categorie_names()

        # self.confusion_mat_name_list_binary = ['Others', self.categorie_names_list[int(self.arg.interest_class) - 1]]
        self.confusion_mat_name_list_multiclass = self.categorie_names_list

        dummy_input = torch.rand(2, self.arg.input_window_size[0], self.arg.input_window_size[1])  # Dummy input to generate a network graph in tensorboard


        self.model = self.load_model(dummy_input = dummy_input)
        self.dev = self.device()

        # Eval the algo's complexity
        # macs, params = profile(self.model, inputs=(dummy_input,))
        #
        # print(f'MACs = {macs/1000000}')

        self.num_class = 27


        self.load_weights()
        self.freeze_layers()

        weight = torch.Tensor(self.arg.binary_weights).to(torch.device(self.dev))

        self.loss_binary = self.load_loss(weight)
        self.loss = self.load_loss(weight=None)


        self.optimizer = self.load_optimizer(self.model)

        self.scheduler = self.load_scheduler()
        self.data_loader = self.load_data()


        self.logger.print_log('Processor: Environment initialized')


    def load_categorie_names(self, ):

        categories_file_path = 'data/categories'

        with open(categories_file_path) as f:
            list_of_actions_names = f.readlines()
        list_of_actions_names = [x.strip() for x in list_of_actions_names]

        f.close()


        return list_of_actions_names

    def load_model(self, dummy_input):
        """
        Load & initialize model specified in configuration
        :return: initialized model
        """

        model = self.fileio.load_model(self.arg.model, **self.arg.model_args, dummy_input=dummy_input)

        return model

    def load_weights(self):
        """
        Load specified weights into model
        """
        if self.arg.weights:
            self.fileio.load_weights(self.model, self.arg.weights, self.arg.ignore_weights)

    def freeze_layers(self, ):
        """
        Freeze the specified layers of the model
        """

        for layer in self.arg.freeze_layers:

            for param in eval("self.model."+layer).parameters():
                param.requires_grad = False

    def load_loss(self, weight):
        """
        Load specified loss
        :return: loss
        """
        if self.arg.loss:
            loss = self.fileio.load_loss(self.arg.loss, weight)
        else:
            loss = None
        return loss

    def load_optimizer(self, model):
        """
        Load specified optimizer
        :return: optimizer
        """
        return self.fileio.load_optimizer(self.arg.optimizer, model, **self.arg.optimizer_args)

    def load_scheduler(self):
        """
        Load specified scheduler
        :return: scheduler
        """
        if self.arg.scheduler:
            scheduler = self.fileio.load_scheduler(self.arg.scheduler, self.optimizer, **self.arg.scheduler_args)
        else:
            scheduler = None
        return scheduler

    def load_data(self,):
        """
        Load data and use specified data feeder and data sampler
        :return:
        """
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        if 'debug' not in self.arg.test_feeder_args:
            self.arg.test_feeder_args['debug'] = self.arg.debug

        # Path to the data directory
        project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(project_directory)
        path_to_data_dir = (project_directory + "/data/Inertial/")
        onlyfiles = [f for f in listdir(path_to_data_dir) if isfile(join(path_to_data_dir, f))]

        subject = []
        train_set = []
        test_set = []

        for ex in onlyfiles:

            sub = re.search('_s(.*)_t', ex)
            subject.append(int(sub.group(1)))

            if int(sub.group(1)) in self.arg.train_subjects:

                train_set.append(ex)

            else:
                test_set.append(ex)



        # Create the training dataloader
        training_generator = create_data_loaders(data_path=path_to_data_dir, list_of_seq=train_set,
                                                 batch_size=self.arg.train_batch_size,
                                                 take_n_examples=self.arg.take_n_examples,
                                                 window=self.arg.input_window_size[0],
                                                 )



        # Create the testing dataloader
        testing_generator = create_data_loaders(data_path=path_to_data_dir, list_of_seq=test_set,
                                                batch_size=self.arg.test_batch_size,
                                                take_n_examples=self.arg.take_n_examples,
                                                window=int(2*self.arg.input_window_size[0]),
                                                )

        data_loader = dict()
        if self.arg.phase == 'train':
            data_loader['train'] = training_generator
            self.logger.print_log(f'DataLoader: {len(data_loader["train"].dataset)} training samples loaded')


        if self.arg.test_feeder_args:
            data_loader['test'] = testing_generator
            self.logger.print_log(f'DataLoader: {len(data_loader["test"].dataset)} test samples loaded')

        return data_loader

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def device(self):
        """
        Set used device: CPU / single GPU
        :return: used devuce
        """
        if self.arg.use_gpu and torch.cuda.device_count():
            dev = "cuda:0"  # single GPU
        else:
            dev = "cpu"

        # move modules to selected device
        self.model = self.model.to(dev)



        return dev

    def start(self):
        """
        Start training of an model
        This function unifies the whole procedure on a very high level with
            * model training
            * model evaluation
            * saving weights
            * parametrization of optimizer / scheduler
            * logging
        """
        self.logger.print_log(f'Parameters:\n{str(vars(self.arg))}\n')

        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.logger.print_log(f'Training epoch: {epoch}')
                self.train()

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    filename = f'epoch{epoch + 1}_model.pt'
                    self.fileio.save_weights(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.logger.print_log(f'Eval epoch: {epoch}')
                    self.test()

                # scheduler
                if self.scheduler:
                    self.scheduler.step()

        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.logger.print_log(f'Model:   {self.arg.model}.')
            self.logger.print_log(f'Weights: {self.arg.weights}.')

            # evaluation
            self.logger.print_log('Evaluation Start:')
            self.test()

    def train(self):
        """
        Train model an epoch
        This function is the real training of the model with
            * forward pass
            * backward pass
            * optimization of weights
            * logging of single iterations
        """

        # Put model in train mode
        self.model.train()

        # Fetch the training data loader
        loader = self.data_loader['train']

        # Batch loss accumulators
        loss_value_bin = []

        # Labels batch accumulator
        label_multiclass_frag = []
        meta_labels_frag = [[] for _ in range(self.num_class)]

        # Predictions batch accumulator
        meta_prediction_frag = [[] for _ in range(self.num_class)]


        with tqdm(total=len(loader)) as t:
            # Todo: enlever binary labels
            for data, label in loader:

                # get data
                data = data.float().to(self.dev)
                label_multiclass = label.long().to(self.dev)

                # Create the meta binary labels
                bin_labels_batch = self.create_meta_binary_labels(label_multiclass, self.num_class)

                # binary_labels = binary_labels.long().to(self.dev)
                label_multiclass_frag.append(label_multiclass.data.cpu().numpy())



                # Forward Pass (inference)
                predictions = self.model(data)
                prediction_multiclass = predictions[0]
                meta_binary_predictions = predictions[1]



                # Logging and sparsing the meta binary predictions and labels
                for i in range(self.num_class):
                    meta_prediction_frag[i].append(meta_binary_predictions[:,
                                                   i*2:(i+1)*2].data.cpu().numpy().argsort()[:,-1])

                    meta_labels_frag[i].append(bin_labels_batch[:,i].data.cpu().numpy())


                # Compute the meta binary loss for the first element
                meta_binary_loss = self.loss_binary(meta_binary_predictions[:, 0:2],
                                                    bin_labels_batch[:, 0])

                # Compute the meta binary loss for the other elements
                for i in range(self.num_class - 1):
                    meta_binary_loss = meta_binary_loss + self.loss_binary(
                        meta_binary_predictions[:, (i + 1) * 2:(i + 2) * 2],
                        bin_labels_batch[:, (i+1)])



                # Batch statistics
                loss_value_bin.append(meta_binary_loss.data.item())


                # Progress bar statistics
                self.iter_info['bin_loss_meta'] = meta_binary_loss.data.item()

                # Update weights
                self.optimizer.zero_grad()
                # loss.backward()
                meta_binary_loss.backward()
                self.optimizer.step()


                # Updating general iteration info
                self.show_iter_info(t)
                self.iter_info['learning rate'] = self.optimizer.param_groups[0]["lr"]
                self.meta_info['iter'] += 1


                ########################### Epoch metrics ############################


            self.epoch_info['mean_loss_train_binary'] = np.mean(loss_value_bin)
            self.result['label_multiclass'] = np.concatenate(label_multiclass_frag)


        # Calculate the precision and recall score for each class from the meta-binary-predictions
        self.train_recall_precision_bin = self.calculate_meta_precision_and_recall(meta_prediction_frag, meta_labels_frag)


        self.show_epoch_info()

        # Reshuffle the data
        self.data_loader = self.load_data()

    def test(self):
        """
        Test model and print out / store statistics
        """
        # Put model in eval mode
        self.model.eval()

        # Fetch the test set dataloader
        loader = self.data_loader['test']

        # Batch loss accumulators
        loss_value_bin = []

        # Labels batch accumulator
        label_multiclass_frag = []
        meta_labels_frag = [[] for _ in range(self.num_class)]

        # Predictions batch accumulator
        multiclass_pred_frag = []
        meta_prediction_frag = [[] for _ in range(self.num_class)]

        P = torch.nn.Softmax(dim=1)
        bin_prob_frag = [[] for _ in range(self.num_class)]

        # Embeding representation
        action_card_frag = []




        with tqdm(total=len(loader)) as t:
            for data, label in loader:


                # get data
                data = data.float().to(self.dev)
                label_multiclass = label.long().to(self.dev)

                # Create the meta binary labels
                bin_labels_batch = self.create_meta_binary_labels(label_multiclass, self.num_class)

                # binary_labels = binary_labels.long().to(self.dev)
                label_multiclass_frag.append(label_multiclass.data.cpu().numpy())


                semantic_info = self.model(data)[1]
                action_card_frag.append(semantic_info.data.cpu().numpy())


                # inference
                with torch.no_grad():
                    predictions = self.model(data)
                    prediction_multiclass = predictions[0]
                    meta_binary_predictions = predictions[1]


                    multiclass_pred_frag.append(prediction_multiclass.data.cpu().numpy())

                    # Logging and sparsing the meta binary predictions and labels
                    for i in range(self.num_class):
                        meta_prediction_frag[i].append(meta_binary_predictions[:,
                                                       i * 2:(i + 1) * 2].data.cpu().numpy().argsort()[:, -1])

                        meta_labels_frag[i].append(bin_labels_batch[:, i].data.cpu().numpy())

                        # The probability of being true from each binary classifiers
                        bin_prob_frag[i].append(P(meta_binary_predictions[:, i * 2:(i + 1) * 2])[:,1])


                    # Compute the meta binary loss for the first element
                    meta_binary_loss = self.loss_binary(meta_binary_predictions[:, 0:2],
                                                        bin_labels_batch[:, 0])

                    # Compute the meta binary loss for the other elements
                    for i in range(self.num_class - 1):
                        meta_binary_loss = meta_binary_loss + self.loss_binary(
                            meta_binary_predictions[:, (i + 1) * 2:(i + 2) * 2],
                            bin_labels_batch[:, i+1])

                    # Batch statistics
                    loss_value_bin.append(meta_binary_loss.data.item())

                    # Progress bar statistics
                    self.iter_info['bin_loss_meta'] = meta_binary_loss.data.item()


                # Updating general iteration info
                self.show_iter_info(t)

                    ########################### Epoch metrics ############################


        predictions = np.concatenate(multiclass_pred_frag)

        for i in range(self.num_class):
            bin_prob_frag[i] = torch.cat(bin_prob_frag[i])

        bin_prob_frag = torch.stack(bin_prob_frag, dim=1)


        # For the interest class
        self.eval_info['mean_loss_test_binary'] = np.mean(loss_value_bin)

        self.result['prediction_multiclass'] = np.concatenate(multiclass_pred_frag).argsort()[:,
                                               ::-1]  # sort with descending probabilities

        self.result['bin_unit_max_prediction'] = bin_prob_frag.data.cpu().numpy().argsort()[:,::-1]

        self.result['label_multiclass'] = np.concatenate(label_multiclass_frag)



        # Calculate the precision and recall score for each class from the meta-binary-predictions
        self.test_recall_precision = self.calculate_meta_precision_and_recall(meta_prediction_frag,
                                                                               meta_labels_frag)


        self.result['action_card'] = np.concatenate(action_card_frag)



        for k in self.arg.show_topk:  # calculate top-k accuracy
            self.eval_info[f'top_{k}_accuracy_bin_max'] = self.calculate_topk(k)

        self.show_eval_info()








    def show_epoch_info(self):
        """
        Show informations per epoch
        """
        for k, v in self.epoch_info.items():
            self.logger.print_log(f'EpochInfo\t{k}: {v}')


        self.logger.tensorboard_log(self.epoch_info, self.meta_info['epoch'])



        self.logger.meta_precision_recall_histogram(recall_precision=self.train_recall_precision_bin,
                                        classes_name_list=self.categorie_names_list,
                                        global_step=self.meta_info['epoch'],
                                        name='TRAIN [Binary] meta_precision_recall_histogram')



    def show_eval_info(self, ):
        """
        Show extended informations after an evaluation / testing phase
        """

        for k, v in self.eval_info.items():
            self.logger.print_log(f'EvalInfo\t{k}: {v}')
        self.logger.tensorboard_log(self.eval_info, self.meta_info['epoch'])


        self.logger.tensorboard_confusion_matrix(name='One_vs_all_Confusion_MATRIX',
                                                 ground_truth=self.result['label_multiclass'],
                                                 prediction=self.result['bin_unit_max_prediction'][:, 0],
                                                 classes=self.confusion_mat_name_list_multiclass,
                                                 global_step=self.meta_info['epoch'])



        self.logger.meta_precision_recall_histogram(recall_precision=self.test_recall_precision,
                                                    classes_name_list=self.categorie_names_list,
                                                    global_step=self.meta_info['epoch'],
                                                    name='TEST [Binary] meta_precision_recall_histogram')



    def show_iter_info(self, progress_bar):
        """
        Show informations per iteration
        :param progress_bar: tqdm progress bar
        """
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            self.logger.tensorboard_log(self.iter_info, self.meta_info['iter'])
        self.update_progress_bar(progress_bar)



    def update_progress_bar(self, progress_bar):
        """
        Update progress bar during training / testing
        :param progress_bar: tqdm progress bar
        """


        progress_bar.set_postfix(loss_bin=f'{self.iter_info["bin_loss_meta"]:05.3f}')

        progress_bar.update()



    def calculate_acc_bin(self, binary_prediction):
        """
        Calculate top-k accuracy
        :param k: k
        :return: accuracy
        """
        # compare label against top-k (=highest k) probabilities
        hit_top_k = [l in [binary_prediction[i]] for i, l in enumerate(self.result['label_binary'])]
        accuracy = sum(hit_top_k) * 100.0 / len(hit_top_k)
        return accuracy

    def calculate_topk(self, k):
        """
        Calculate top-k accuracy
        :param k: k
        :return: accuracy
        """
        # compare label against top-k (=highest k) probabilities
        hit_top_k_agg = [l in self.result['bin_unit_max_prediction'][i, :k] for i, l in enumerate(self.result['label_multiclass'])]
        bin_max_accuracy = sum(hit_top_k_agg) * 100.0 / len(hit_top_k_agg)



        return bin_max_accuracy


    def calculate_recall(self, binary_predictions, binary_labels):
        """
        Calculate recall (true_positive / (true_pos + False_neg)

        :return: Recall ratio
        """
        true_positive = 0
        false_negative = 0
        for i, pred in enumerate(binary_predictions):

            if pred == binary_labels[i] and pred == 1:
                true_positive += 1

            if pred != binary_labels[i] and pred == 0:
                false_negative += 1

        recall = true_positive/(true_positive+false_negative+1)
        return recall


    def calculate_precision(self, binary_predictions, binary_labels):
        """
        Calculate Precision (true_positive / (true_pos + false_pos)
        :return: precision
        """

        true_positive = 0
        false_positive = 0
        for i, pred in enumerate(binary_predictions):

            if pred == binary_labels[i] and pred == 1:
                true_positive += 1

            if pred != binary_labels[i] and pred == 1:
                false_positive +=1

        precision = true_positive/(true_positive+false_positive+1)
        return precision


    def calculate_meta_precision_and_recall(self, meta_prediction_frag, meta_labels_frag):
            # Meta-evaluation
            recall_precision = []
            epoch_meta_prediction = [[] for _ in range(self.num_class)]
            epoch_meta_labels = [[] for _ in range(self.num_class)]
            for i in range(self.num_class):
                # Concatenate the prediction of all the batches

                # Create arrays with the predictions and labels for all the binary classes combining all the batches
                epoch_meta_prediction[i] = np.concatenate(meta_prediction_frag[i])
                epoch_meta_labels[i] = np.concatenate(meta_labels_frag[i])

                # Find the index of the positive and negative labels for every class
                # Positives are 1 and Negatives are 0
                positive_labels_idx = np.where(epoch_meta_labels[i] == 1)
                negative_labels_idx = np.where(epoch_meta_labels[i] == 0)

                nb_true_positive = np.sum(np.equal(epoch_meta_prediction[i][positive_labels_idx],
                                                   epoch_meta_labels[i][positive_labels_idx]))

                nb_false_positive = np.sum(
                    np.not_equal(epoch_meta_prediction[i][negative_labels_idx],
                                 epoch_meta_labels[i][negative_labels_idx]))

                # Compute the Recall and Precision scores
                recall = 100 * nb_true_positive / len(positive_labels_idx[0])
                precision = 100*nb_true_positive/(nb_true_positive + nb_false_positive+1)
                recall_precision.append([recall, precision])

            return np.array(recall_precision)

    def create_meta_binary_labels(self, label_multiclass, num_class):
        for k, label in enumerate(label_multiclass):
            bin_label = torch.zeros([num_class])

            for i in range(num_class):

                if (label == i):
                    bin_label[i] = 1


            if k == 0:
                bin_labels_batch = bin_label.unsqueeze(dim=0)

            else:
                bin_labels_batch = torch.cat((bin_labels_batch, bin_label.unsqueeze(dim=0)), dim=0)

        return bin_labels_batch.long().to(self.dev)

    @staticmethod
    def get_parser(add_help=True):
        """
        Extended argument parser with general options for the processor
        :param add_help: boolean flag to enable command line help
        :return: parser
        """
        # parameter priority: command line > config > default
        parser = super(Processor, Processor).get_parser()
        parser.description = 'Processor'
        parser.add_argument('--use_gpu', action='store_true', default=False, help='use GPUs or not')
        parser.add_argument('--debug', action="store_true", default=False, help='less data, faster loading')

        # processor
        parser.add_argument('--phase', default='train', help='train or test')
        parser.add_argument('--save_result', action="store_true", default=False, help='save output of model')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='indexes of GPUs for training or testing')

        # visualize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', action="store_true", default=True, help='save logging or not')
        parser.add_argument('--print_log', action="store_true", default=True, help='print logging or not')
        parser.add_argument('--show_topk', type=int, default=[1, 2, 5], nargs='+', help='show top-k accuracies')

        # model
        parser.add_argument('--model', default=None, help='type of model')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='arguments for model')
        parser.add_argument('--weights', default=None, help='weights for model initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='ignored weights during initialization')
        parser.add_argument('--loss', default=None, help='type of loss function')
        parser.add_argument('--freeze_layers', type=str, default=[], nargs='+',
                            help='freeze these layers during initialization')

        # optimizer
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--optimizer_args', action=DictAction, default=dict(), help='arguments for optimizer')
        parser.add_argument('--binary_weights', type=list, default=[0.1, 1], help='The size of the input')

        # scheduler
        parser.add_argument('--scheduler', default=None, help='type of scheduler')
        parser.add_argument('--scheduler_args', action=DictAction, default=dict(), help='arguments for scheduler')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='type of data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='arguments for training data loader')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='arguments for test data loader')
        parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training')
        parser.add_argument('--test_batch_size', type=int, default=256, help='batch size for test')
        parser.add_argument('--num_worker', type=int, default=4, help='number of workers per gpu for data loader')
        parser.add_argument('--sampler', type=str, default=None, help='type of sampler')
        parser.add_argument('--weight_function', type=str, default='equal', help='Choose how to compute the sampling weights')

        # Dataloader
        parser.add_argument('--take_n_examples', type=int, default=-1, help='Choose a certain amount of ex. for debug')
        parser.add_argument('--train_subjects', type=list, default=[1,3,5,7], help='List of the subjects used for training')

        # Pre-treatment
        parser.add_argument('--input_window_size', type = list, default=[326, 6], help='The size of the input')


        return parser


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]