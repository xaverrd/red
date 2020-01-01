from numpy import dot, add, subtract, multiply, divide, array, append, vectorize, zeros, save, load, sum, repeat, amax
from numpy.random import randn
from os import mkdir
from os.path import join
from red.operator import *



class Network:



    def create(self, layers):
        self.layers = layers
        self.weights = [0]
        self.biases = [0]
        for num_layer in range(1, len(layers)):
            self.weights.append(randn(layers[num_layer], layers[num_layer-1]))
            self.biases.append(zeros(layers[num_layer]))



    def keep(self, path):
        self.path = path
        try:
            mkdir(join(path, 'Network'))
        except FileExistsError:
            pass
        save(join(path, 'Network')+'/layers', self.layers)
        save(join(path, 'Network')+'/weights', self.weights)
        save(join(path, 'Network')+'/biases', self.biases)



    def update(self):
        save(join(self.path, 'Network')+'/layers', self.layers)
        save(join(self.path, 'Network')+'/weights', self.weights)
        save(join(self.path, 'Network')+'/biases', self.biases)



    def retrieve(self, path):
        self.layers = load(join(path, 'Network')+'/layers.npy', allow_pickle=True).tolist()
        self.weights = load(join(path, 'Network')+'/weights.npy', allow_pickle=True).tolist()
        self.biases = load(join(path, 'Network')+'/biases.npy', allow_pickle=True).tolist()



    def fire(self, input):
        weighted_sums = [0]
        activations = [input]
        for num_layer in range(1, len(self.layers)):
            weighted_sums.append(add(dot(self.weights[num_layer], activations[num_layer-1]), self.biases[num_layer]))
            activations.append(sigmoid(weighted_sums[num_layer]))
        return weighted_sums, activations



    def predict(self, input, expected, types):
        weighted_sums, activations = self.fire(input)
        loss = sum(squared_difference(activations[-1], expected))
        return weighted_sums, activations, loss



    def correct(self, input, expected, types):
        weighted_sums, activations, loss = self.predict(input, expected, types)
        errors = [0]*(len(self.layers)-1) + [multiply(sigmoid_slope(weighted_sums[-1]), squared_difference_slope(activations[-1], expected))]
        weight_corrections = [0]*(len(self.layers)-1) + [repeat([array([error]) for error in errors[-1]], self.layers[-2], axis=1)]
        bias_corrections = [0]*(len(self.layers)-1) + [errors[-1]]
        for num_layer in range(len(self.layers)-2, 0, -1):
            errors[num_layer] = zeros(self.layers[num_layer])
            for num_node in range(self.layers[num_layer+1]):
                errors[num_layer] += errors[num_layer+1][num_node] * self.weights[num_layer+1][num_node]
            errors[num_layer] *= sigmoid_slope(weighted_sums[num_layer])
            weight_corrections[num_layer] = repeat([array([error]) for error in errors[num_layer]], self.layers[num_layer-1], axis=1)
            bias_corrections[num_layer] = errors[num_layer]
        for num_layer in range(1, len(self.layers)):
            weight_corrections[num_layer] *= activations[num_layer-1]
        return weight_corrections, bias_corrections, loss



    def train(self, epochs, batch_size, data, labels, types, learning_rate):
        for epoch in range(epochs):
            data, labels = shuffle(data, labels)
            items_total = len(data)
            items_index = 0
            print('Shuffled...')
            while True:
                weight_corrections = zeros(len(self.layers))
                bias_corrections = zeros(len(self.layers))
                loss = 0
                if items_total-items_index > batch_size:
                    for i in range(batch_size):
                        iteration_weight_corrections, iteration_bias_corrections, iteration_loss = self.correct(data[items_index], labels[items_index], types)
                        weight_corrections = (array(weight_corrections) + (array(iteration_weight_corrections)*learning_rate/batch_size)).tolist()
                        bias_corrections = (array(bias_corrections) + (array(iteration_bias_corrections)*learning_rate/batch_size)).tolist()
                        loss += iteration_loss/batch_size
                        items_index += 1
                else:
                    for i in range(items_left):
                        iteration_weight_corrections, iteration_bias_corrections, iteration_loss = self.correct(data[items_index], labels[items_index], types)
                        weight_corrections = add(weight_corrections, iteration_weight_corrections/items_left)
                        bias_corrections = add(bias_corrections, iteration_bias_corrections/items_left)
                        loss += iteration_loss/items_left
                    break
                    #^^^^ break is misplaced
                for num_layer in range(1, len(self.layers)):
                    self.weights[num_layer] -= weight_corrections[num_layer]
                    self.biases[num_layer] -= bias_corrections[num_layer]
                self.update()
                print(loss)
