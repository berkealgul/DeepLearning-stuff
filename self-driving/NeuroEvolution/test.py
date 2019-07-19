from NeuroEvolution.matrix import Matrix
from NeuroEvolution.neuralNetwork import NeuralNetwork
import NeuroEvolution.jsonHandler as jsonh

nn = NeuralNetwork(1,1,1,1)

jsonh.save(nn, 'xd')

