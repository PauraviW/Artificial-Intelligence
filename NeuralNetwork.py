import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.weights = {}
        self.bias = {}
        self.rms_prop_weights = {}
        self.rms_prop_bias = {}
        self.total_layers = 0
        self.batch_size = -1
        self.learning_rate = 0.01
        self.rms_prop_optimizer = False
        self.rho_value = 0.9
        self.activations = {}
        self.supported_activations = {'softmax': self.softmax, 'relu': self.relu, 'sigmoid': self.sigmoid}
        self.supported_activations_derivatives = {'sigmoid': self.sigmoid_derivative, 'relu': self.relu_derivative}
        self.accuracy = []

    def add(self, shape, activation):
        self.weights[self.total_layers] = np.random.randn(shape[0], shape[1])  # * np.sqrt(2 / shape[0] + shape[1])
        self.rms_prop_weights[self.total_layers] = np.zeros((shape[0], shape[1]))
        self.bias[self.total_layers] = np.zeros((1, shape[1]))
        self.rms_prop_bias[self.total_layers] = np.zeros((1, shape[1]))
        self.activations[self.total_layers] = self.supported_activations[activation]
        self.total_layers += 1

    def softmax(self, x):
        expA = np.exp(x - np.amax(x, axis=1, keepdims=True))
        return expA / expA.sum(axis=1, keepdims=True)

    def softmax_derivative(self, x):
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def relu(self, x):
        x[x < 0] = 0
        return x

    def relu_derivative(self, x):
        return (x > 0).astype(int)

    # One-hot-encoding needed
    def cross_entropy_loss(self, y_actual, y_pred):
        return -1 * np.sum(y_actual * np.log(y_pred))

    # derivative of cross entropy loss wrt softmax input
    def cross_entropy_loss_gradient(self, y_actual, y_pred):
        return y_pred - y_actual

    def predict(self, x):
        post_activation_previous_layer_output = x
        for layer in range(self.total_layers):
            pre_activation_output = np.dot(post_activation_previous_layer_output, self.weights[layer]) + \
                                    self.bias[layer]
            post_activation_previous_layer_output = self.activations[layer](pre_activation_output)
        return post_activation_previous_layer_output

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        return sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y_pred)

    def fit(self, x, y, batch_size=None, epoch=None, learning_rate=None, rms_prop_optimizer=False, rho_value=None):

        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = len(x)

        if epoch is None:
            epoch = 1

        if rms_prop_optimizer:
            self.rms_prop_optimizer = rms_prop_optimizer
            self.rho_value = rho_value

        if learning_rate is not None:
            self.learning_rate = learning_rate
        for i in range(epoch):
            for batch in range(0, len(x), self.batch_size):
                x_batch = x[batch:batch + self.batch_size]
                y_batch = y[batch:batch + self.batch_size]

                # Forward Propagation
                pre_activation_outputs, post_activation_outputs = self.forward_propagation(x_batch)

                # Back Propagation
                delta_bias, delta_weights = self.back_propagation(y_batch, post_activation_outputs, pre_activation_outputs)

                # Update Weights and Bias
                self.update_weights_and_bias(delta_weights, delta_bias)

    # Gradient Descent Update
    def update_weights_and_bias(self, delta_weights, delta_bias):
        if self.rms_prop_optimizer:
            for layer in range(self.total_layers):
                self.rms_prop_bias[layer] = self.rho_value * self.rms_prop_bias[layer] + (1 - self.rho_value) * \
                                            (delta_bias[layer] ** 2)
                self.bias[layer] = self.bias[layer] - self.learning_rate * (delta_bias[layer] /
                                                                            (np.sqrt(self.rms_prop_bias[layer]) + 1e-8))

                self.rms_prop_weights[layer] = self.rho_value * self.rms_prop_weights[layer] + (1 - self.rho_value) * \
                                               (delta_weights[layer] ** 2)
                self.weights[layer] = self.weights[layer] - self.learning_rate * (delta_weights[layer] /
                                                                        (np.sqrt(self.rms_prop_weights[layer]) + 1e-8))
        else:
            for layer in range(self.total_layers):
                self.bias[layer] = self.bias[layer] - self.learning_rate * delta_bias[layer]
                self.weights[layer] = self.weights[layer] - self.learning_rate * delta_weights[layer]

    def forward_propagation(self, x):
        pre_activation_outputs = {}
        post_activation_previous_layer_output = x
        post_activation_outputs = {-1: x}

        for layer in range(self.total_layers):
            pre_activation_outputs[layer] = np.dot(post_activation_previous_layer_output, self.weights[layer]) + \
                                            self.bias[layer]
            post_activation_previous_layer_output = self.activations[layer](pre_activation_outputs[layer])
            post_activation_outputs[layer] = post_activation_previous_layer_output

        return pre_activation_outputs, post_activation_outputs

    def back_propagation(self, y, post_activation_outputs, pre_activation_outputs):
        delta_bias = {}
        delta_weights = {}
        # Last Layer delta_weights and delta_bias calculations
        loss = self.cross_entropy_loss_gradient(y, post_activation_outputs[self.total_layers - 1])
        delta_bias[self.total_layers - 1] = loss
        delta_weights[self.total_layers - 1] = np.dot(post_activation_outputs[self.total_layers - 2].T, loss)

        for layer in range(self.total_layers - 2, -1, -1):
            delta_bias[layer] = np.dot(delta_bias[layer + 1], self.weights[layer + 1].T) * \
                                self.supported_activations_derivatives[self.activations[layer].__name__](
                                    pre_activation_outputs[layer])

            delta_weights[layer] = np.dot(post_activation_outputs[layer - 1].T, delta_bias[layer])

        for k in delta_bias.keys():
            delta_bias[k] = np.sum(delta_bias[k], axis=0, keepdims=True)

        return delta_bias, delta_weights

    def save(self, fname):
        model_variables = locals()['self']
        f = open('trained_models/nnet/' + fname, 'w')
        f.write('total_layers=%d\n' % self.total_layers)
        f.write('learning_rate=%f\n' % self.learning_rate)
        for layer in range(self.total_layers):
            f.write('layer=%d\n' % layer)
            f.write('Activation_Functions=%s\n' % model_variables.activations[layer].__name__)
            f.write('layer_%s_weights_file=nn_layer_%s_weights.txt\n' % (layer, layer))
            f.write('layer_%s_bias_file=nn_layer_%s_bias.txt\n' % (layer, layer))
            np.savetxt('trained_models/nnet/nn_layer_%s_weights.txt' % layer, model_variables.weights[layer])
            np.savetxt('trained_models/nnet/nn_layer_%s_bias.txt' % layer, model_variables.bias[layer])
        f.close()

    def load(self, fname):
        f = open('trained_models/nnet/' + fname, 'r')
        model = f.readlines()
        f.close()
        layer = 0
        self.total_layers = int(model[0].strip().split('=')[1])
        self.learning_rate = float(model[1].strip().split('=')[1])
        for i in range(2, len(model), 4):
            self.activations[layer] = self.supported_activations[model[i + 1].strip().split('=')[1]]
            self.weights[layer] = np.loadtxt('trained_models/nnet/' + model[i + 2].split('=')[1].strip())
            self.bias[layer] = np.loadtxt('trained_models/nnet/' + model[i + 3].split('=')[1].strip())
            layer += 1
