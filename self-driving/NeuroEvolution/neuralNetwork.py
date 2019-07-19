from NeuroEvolution.matrix import Matrix
import math
import random as rnd
import copy


class NeuralNetwork:
    def __init__(self, inputNeuronC=1, outputNeuronC=1, hiddenNeuronPerLayerC=0, hiddenLayerC=0):

        # BU ALAN DENEYSELDİR DEĞİŞEBİLİR

        # hiç parametre verilmemiş ise hiç birşey yapmiyacak
        # amacı ise json verisinden yüklenmek istenirse boşuna başlangıç
        # verileri oluşturulmasın
        if inputNeuronC + outputNeuronC + hiddenNeuronPerLayerC + hiddenLayerC == 2:
            self.inputsForTrain = []
            self.outputsForTrain = []
            return

        # parametre oygunluğu kontrol edilir
        if inputNeuronC < 1 or outputNeuronC < 1 or hiddenNeuronPerLayerC < 0 or hiddenLayerC < 0:
            raise Exception('çıktı ve girdi nöronları 1, diğeri 0 dan küçük olamaz')

        # BU ALAN DENEYSELDİR DEĞİŞEBİLİR

        # kontrol sonrası oluşturma işlemleri başlatılır

        self.weights = []
        self.biases = []

        # sgd + momentum optimizasyonu için
        # ağırlık ve sapmaların değişim hızlarını kayıt ediyoruz
        # burdaki matrislerin boyutları karşılık gelen ağırlık-sapmalarıyla aynı olacak
        self.Vweight = []
        self.Vbias = []

        # gradient decent metotdu ile eğitirken her katmanın girdi(input)
        # ve çıktıları(output) lazım olacak bu yüzden bunları kayıt altına
        # alıp güncellemek için özel listeler hazırlıyoruz
        self.inputsForTrain = []
        self.outputsForTrain = []

        # ağırlık ve sapmalar verilen kurala görre ayarlanır:
        # weight = Matrix(next_layer_neuronC, previus_layer_neuronC)
        # bias = Matrix(next_layer_neuronC, 1)

        # input - hidden
        self.weights.append(Matrix(hiddenNeuronPerLayerC, inputNeuronC))
        self.biases.append(Matrix(hiddenNeuronPerLayerC, 1))

        self.Vweight.append(Matrix(hiddenNeuronPerLayerC, inputNeuronC))
        self.Vbias.append(Matrix(hiddenNeuronPerLayerC, 1))

        # hidden - hidden
        for i in range(hiddenLayerC - 1):
            # kaç saklı katman varsa onun 1 eksiği kadar aralarında ağırlık ve sapma vardır
            self.weights.append(Matrix(hiddenNeuronPerLayerC, hiddenNeuronPerLayerC))
            self.biases.append(Matrix(hiddenNeuronPerLayerC, 1))

            self.Vweight.append(Matrix(hiddenNeuronPerLayerC, hiddenNeuronPerLayerC))
            self.Vbias.append(Matrix(hiddenNeuronPerLayerC, 1))

        # hidden  - output
        self.weights.append(Matrix(outputNeuronC, hiddenNeuronPerLayerC))
        self.biases.append(Matrix(outputNeuronC, 1))

        self.Vweight.append(Matrix(outputNeuronC, hiddenNeuronPerLayerC))
        self.Vbias.append(Matrix(outputNeuronC, 1))

        for i in range(hiddenLayerC + 1):
            # nöron ağdaki bütün katman sayısı kadar "inputsForTrain" ve "outputsForTrain"
            # değişkenlerine null mabında 0 değerini veriyoruz

            # not:
            # zaten feedforward sürecinde bu değişkenler değişeceği için 0 olması hataya sebep olmayacak
            # lakin indislerle işlem yapabilmek adına listede hazırda eleman olması gerekir
            self.inputsForTrain.append(0)
            self.outputsForTrain.append(0)

        self.randomize_weight_and_biases()

    def feedforward(self, input):
        output = input

        for i in range(len(self.weights)):
            # işlemleri gerçekleştirmeden önce girdiyi kaydediyoruz
            # output demesine bakma bir önceki katmanın çıtkısı bi sonrakinin girdisidir
            # eğitim sırasında girdinin transpoz edilmiş hali lazım olacak (backpropagation fonksiyonuna bakabilirsin)
            self.inputsForTrain[i] = output.transpose()

            # girdi i olmak üzere(burda output olarak isimlendirdim) f((i * w) + b) işlemi
            # gerçekleştirilir
            output = Matrix.matrix_product(self.weights[i], output)
            output.add(self.biases[i])

            # çıtkıları aktive ediyoruz ki çıtkı değerleri 0-1 arasında olsun
            for y in range(output.rowCount):
                for x in range(output.columnCount):
                    output.values[y][x] = NeuralNetwork.activation(output.values[y][x])

            # yeni katmana geçmeden önce çıtkıyı kaydediyoruz
            self.outputsForTrain[i] = output.copy()

        # sinir ağından çıkan son veriyi dönderiyoruz
        return output

    def backpropagation(self, lr, friction, error):
        # sondan başa gittiğimiz için en son ağırlıklardan ve önyargılardan başlayacağiız
        i = len(self.weights) - 1
        while i > -1:
            o = self.outputsForTrain[i]
            i_t = self.inputsForTrain[i]

            # sapmanın gradyanı = gradyan = compute_gradient()
            # ağırlığın gradyanı = gradyan x i_t
            gradient = NeuralNetwork.compute_gradient(lr, error.copy(), o)

            # ağırlıklar ve sapmalar gradyan(sistem ivmesi) bazında hesaplanır
            NeuralNetwork.momentum_update(self.Vbias[i], self.biases[i], gradient, friction)

            gradient = Matrix.matrix_product(gradient, i_t)
            NeuralNetwork.momentum_update(self.Vweight[i], self.weights[i], gradient, friction)

            # bir sonraki katman için hataları hazırlıyoruz
            w_t = self.weights[i].transpose()
            error = Matrix.matrix_product(w_t, error)

            i -= 1

    def randomize_weight_and_biases(self):
        for w in self.weights:
            NeuralNetwork.randomize_matrix(w)
        for b in self.biases:
            NeuralNetwork.randomize_matrix(b)

    @staticmethod
    def calculate_error(target, output):
        # e = (t - o)
        error = target.copy()
        error.sub(output)
        return error

    @staticmethod
    def activation(x):
        # sigmoid fonksiyonu
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def compute_gradient(lr, e, o):
        # gradient = -lr * e * o * (1 - o)

        # e
        gradient = e
        # e * -lr
        gradient.multiply(-lr)
        # e * -lr * o
        gradient.multiply(o)
        # -o
        o.multiply(-1)
        # (1 - o)
        o.add(1)
        # e * -lr * o * (1 - o)
        gradient.multiply(o)

        return gradient

    @staticmethod
    def momentum_update(v, x, gradient, friction):
        # gradient = sistem ivmesi v = sistem hızı x = sistem konumu şeklinde denebilir
        # v'den dolayı ağırlık ve sapmaların gereksiz ilerlemesini engellemek için
        # yani bi bakıma v'yi kontrol altına almak için azaltıcı faktörle(friction) çarparız

        # v = friction * v - gradient
        # x += v # x ağırlık veya sapma olabilir
        v.multiply(friction)
        v.sub(gradient)
        x.add(v)

    @staticmethod
    def randomize_matrix(matrix):
        for y in range(matrix.rowCount):
            for x in range(matrix.columnCount):
                matrix.values[y][x] = rnd.uniform(-1, 1)

    @staticmethod
    def calculate_cost(error):
        # cost = (error^2) / 2
        error.multiply(error)
        error.multiply(0.5)
        return error

    # genetik algoritma için fonksiyon eklentisi
    def copy(self):
        return copy.deepcopy(self)

    def mutate(self, mutationRate):
        for w in self.weights:
            # ağırlık matrisine eriştik
            for row in w.values:
                # matrisin sırasina eriştik
                for i in range(len(row)):
                    # ağırlıklara eriştik
                    if rnd.random() < mutationRate:
                        treshold = rnd.random()
                        row[i] += rnd.uniform(-treshold, treshold)
                        if not -1 < row[i] < 1:
                            row[i] = rnd.uniform(-1, 1)
