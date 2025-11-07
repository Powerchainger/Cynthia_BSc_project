#   Wrapper around the LSTM, it adds functionality for:
#       - loading data from csv files
#       - data preprocessing
#       - preparing the input data for the model

import torch
import numpy as np

from .model import Model

from modules.io.csv_config import Csv_config
from modules.io.load import load_data_from_csv

#TODO: find out if thi can be done differently, it's ugly
from modules.data_preprocessing.preprocessing import pre_process_data

class Model_wrapper():
    # the wrapper needs the model, its params, and a csv file and a config for
    # the csv file in order to be able to initialize the data and input matrix
    def __init__(self, model, model_params, csv_file_path, csv_config):

        #private fields:
        self.__look_back = model_params.look_back # Needed for input matrix
        self.__csv_config = csv_config

        # protected fields:
        self._model = model
        self._csv_file_path = csv_file_path

    def __init_data(self):
        # loading the csv could fail
        try:
            data_raw = load_data_from_csv(self._csv_file_path, self.__csv_config)
        except:
            print(f'ERROR: Could not load:{csv_file_path}')
            exit()

        self.__samples, self._output_scaler = pre_process_data(data_raw)

    def __prepare_sequence(self, idx):

        # assert that we are actually in range of our data
        assert (idx >= self.__look_back) 
        assert (idx < len(self.__samples))
            
        start = idx - self.__look_back
        end = idx

        # sequence is just a list of samples of size look_back
        sequence = [ x for x in self.__samples[start:end] ]

        # target is the actual usage at time idx
        target = [self.__samples[idx][0]] 

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
    def __init_input_matrix(self):
        inputs = []
        targets = []

        # prepare sequences of size look_back from the dataset
        for idx in range(self.__look_back, len(self.__samples)):
            sequence, target = self.__prepare_sequence(idx)
            inputs.append(sequence)
            targets.append(target)

        # set the input matrixs and its targets in the class
        # so that they can be accessed when run_model is called
        # np.array(inputs) and np.array(targets) improves performance when
        # converting the input and targets to tensors
        self._input_matrix = torch.tensor(np.array(inputs))
        self._targets = torch.tensor(np.array(targets))

    # must be implemented by subclass if they want to call run
    def _run_model(self):
        raise NotImplementedError()

    def run(self):    
        # load and prepare the data, such that the model can be run
        self.__init_data()
        self.__init_input_matrix()
   
        self._run_model()

