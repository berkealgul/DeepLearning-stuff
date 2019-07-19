from NeuroEvolution.matrix import Matrix
from NeuroEvolution.neuralNetwork import NeuralNetwork
import json


# ana kayıt fonksiyonu
def save(neuralNet, fileName):
    data = convert_to_json(neuralNet)
    with open(fileName + '.json', 'w') as json_file:
        json.dump(data, json_file)


# yardımcı kayıt fonksiyonlar
def convert_to_json(arg):
    ta = type(arg)

    if ta is NeuralNetwork:
        data = dict()
        data['weights'] = convert_to_json(arg.weights)
        data['biases'] = convert_to_json(arg.biases)
        data['Vweight'] = convert_to_json(arg.Vweight)
        data['Vbias'] = convert_to_json(arg.Vbias)
    elif ta is list:
        data = []
        for element in arg:
           data.append(convert_to_json(element))
    elif ta is Matrix:
        data = convert_matrix_to_json(arg)

    return data


def convert_matrix_to_json(matrix):
    data = dict()
    data['rows'] = matrix.rowCount
    data['columns'] = matrix.columnCount
    data['values'] = matrix.values
    return data


def load_nn(fileName):
    with open(fileName + '.json', 'r') as json_file:
        data = json.load(json_file)
        nn = NeuralNetwork()

        ws = []
        for w in data['weights']:
            ws.append(load_matrix(w))
        nn.weights = ws

        bs = []
        for b in data['biases']:
            bs.append(load_matrix(b))
        nn.biases = bs

        vws = []
        for vw in data['Vweight']:
            bs.append(load_matrix(vw))
        nn.Vweight = vws

        vbs = []
        for vb in data['Vbias']:
            vbs.append(load_matrix(vb))
        nn.Vbias = vbs

        for i in range(len(nn.weights)):
            nn.inputsForTrain.append(0)
            nn.outputsForTrain.append(0)
        return nn


# ana yükleme fonksiyonu
def load_matrix(matrix):
    r = matrix['rows']
    c = matrix['columns']
    values = matrix['values']
    m = Matrix(r, c)
    m.values = values
    return m
