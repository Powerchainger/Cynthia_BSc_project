#   Wrapper around the LSTM, it adds functionality for:
#       - loading data from csv files
#       - data preprocessing
#       - preparing the input data for the model

import torch
import numpy as np

from .model import Model

from modules.io.csv_config import Csv_config
from modules.io.load import load_data_from_csv

#TODO: find out if this can be done differently, it's ugly
from modules.data_preprocessing.preprocessing import pre_process_data

#TODO: make it so the target output is both normalized and un-normalized 

class Model_wrapper():
    # the wrapper needs the model, its params, and a csv file and a config for
    # the csv file in order to be able to initialize the data and input matrix
    def __init__(self, model, model_params, csv_config, eval_file_path, train_file_path=None):

        #private fields:
        self.__look_back = model_params.look_back # Needed for input matrix
        self.__csv_config = csv_config

        self.__samples = {} 

        # protected fields:
        self._model = model
        self._train_file_path = train_file_path
        self._eval_file_path = eval_file_path
        
        self._output_scaler = {} 
        self._input_matrix = {} 
        self._targets = {} 
        self._targets_raw = {} 

    def __init_data(self, phase):
        
        if phase == 'train' :
            file_path = self._train_file_path
        else:
            file_path = self._eval_file_path

        # loading the data could fail
        try:
            data_raw = load_data_from_csv(file_path, self.__csv_config)
        except:
            print(f'ERROR: Could not load:{file_path}')
            exit()

        # pre process the training dataset and validation dataset

        self.__samples[phase], self._output_scaler[phase] = pre_process_data(data_raw)

    def __prepare_sequence(self, phase, idx):

        # assert that we are actually in range of our data
        samples = self.__samples[phase]

        assert (idx >= self.__look_back) 
        assert (idx < len(samples))
            
        start = idx - self.__look_back
        end = idx

        # sequence is just a list of samples of size look_back
        sequence = [ x for x in samples[start:end] ]

        # target is the actual usage at time idx
        target = [samples[idx][0]]

        return (sequence, target)
    
    # The input is constructed as follows:
    # input_sequence = { E', I', D', H'}, where: 
    #   - E' is the sequence of energy consumptions for lookBack time steps
    #   - I' is the corresponding time day indices for lookBack time steps
    #   - D' is the corresponding day of week indices for look_back steps
    #   - H' is the corresponding holiday markers for look_back steps
    #   
    #   - To get E' we normalize E to fit the range [0..1]
    #   - I' D' H' are encoded by a one hot encoder
    def __init_input_matrix(self, phase):
        inputs = []
        targets = []

        samples = self.__samples[phase]

        # prepare sequences of size look_back from the dataset
        for idx in range(self.__look_back, len(samples)):
            sequence, target = self.__prepare_sequence(phase, idx)
            inputs.append(sequence)
            targets.append(target)

        # set the input matrixs and its targets in the class
        # so that they can be accessed when run_model is called
        # np.array(inputs) and np.array(targets) improves performance when
        # converting the input and targets to tensors
        self._input_matrix[phase] = torch.tensor(np.array(inputs))
        self._targets[phase] = torch.tensor(np.array(targets))
        #TODO: do this somewhere else, this is double work
        self._targets_raw[phase] = self._output_scaler[phase].inverse_transform(
            self._targets[phase].detach().clone().numpy())


    # must be implemented by subclass if they want to call run
    def _run_model(self):
        raise NotImplementedError()

    # must be implemented by subclass if they want to call train
    def _train_model(self) :
        raise NotImplementedError()
    
    def run(self):    
        # load and prepare the data, such that the model can be run

        # check that we have file paths for evaluation
        assert(self._eval_file_path != None)

        self.__init_data('eval')
        self.__init_input_matrix('eval')
  
        self._run_model()

    def train(self):
        # load and prepare the data, such that the model can be trained

        # check that we have file paths for both training and evaluation
        assert(self._train_file_path != None)
        assert(self._eval_file_path != None)

        self.__init_data('train')
        self.__init_input_matrix('train')

        self.__init_data('eval')
        self.__init_input_matrix('eval')
        
        self._train_model()




