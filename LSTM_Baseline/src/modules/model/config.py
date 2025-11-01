#POD containing the parameters for the LSTM
class Model_config():
    def __init__(
        self,
        inputDim,
        hiddenDim,
        layerDim,
        outputDim,
        lookBack,
        lossFunction,
        optimizer,
        epochs):

        self.inputDim = inputDim 
        self.hiddenDim = hiddenDim
        self.layerDim = layerDim
        self.outputDim = outputDim
        self.lookBack = lookBack
        self.lossFunction = lossFunction
        self.optimizer = optimizer
        self.epochs = epochs
