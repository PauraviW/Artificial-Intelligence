import numpy as np
from scipy import stats


class KNearestNeighbor:
    def __init__(self, no_of_neighbors=10, strategy='euclidean'):
        self.no_of_neighbors = no_of_neighbors
        self.batch_size = -1
        self.train_x = None
        self.train_y = None
        self.current_strategy = 'euclidean'
        self.supported_strategies = {'euclidean': self.calculate_euclidean_distance}

    def fit(self, x, y, batch_size=None):
        self.train_x = x
        self.train_y = y
        if batch_size is not None:
            self.batch_size = batch_size

    def predict(self, x):
        K = self.no_of_neighbors
        accuracy = []
        pred_y = np.zeros(shape=(1, 1))

        array_equal = False
        if np.array_equal(x, self.train_x):
            array_equal = True

        if self.batch_size != -1:
            batch_size = self.batch_size
        else:
            batch_size = len(x)
        for batch in range(0, len(x), batch_size):
            dist_sq = self.supported_strategies[self.current_strategy](x[batch:batch + batch_size], self.train_x)
            y_actual = self.train_y[batch:batch + batch_size, np.newaxis]
            nearest_neighbors = None
            if array_equal:
                nearest_neighbors = np.argsort(dist_sq, axis=1)[:, 1:K + 1]
            else:
                nearest_neighbors = np.argsort(dist_sq, axis=1)[:, 0:K]

            batch_pred_y = stats.mode(self.train_y[nearest_neighbors], axis=1)[0]
            pred_y = np.append(pred_y, batch_pred_y.reshape(batch_pred_y.shape[0], 1), axis=0)
        return pred_y[1:, :]

    def score(self, x, y):
        pred_y = self.predict(x)
        return np.count_nonzero(y == pred_y) / len(pred_y)

    def save(self, fname):
        model_variables = locals()['self']
        f = open('trained_models/nearest/' + fname, 'w')
        f.write('batch_size=%d\n' % model_variables.batch_size)
        f.write('current_strategy=%s\n' % model_variables.current_strategy)
        f.write('no_of_neighbors=%d\n' % model_variables.no_of_neighbors)
        f.write('train_x_file=knn_train_x.txt\n')
        f.write('train_y_file=knn_train_y.txt\n')
        f.close()
        np.savetxt('trained_models/nearest/knn_train_x.txt', model_variables.train_x)
        np.savetxt('trained_models/nearest/knn_train_y.txt', model_variables.train_y)

    def load(self, fname):
        f = open('trained_models/nearest/' + fname, 'r')
        model = f.readlines()
        f.close()
        self.batch_size = int(model[0].split('=')[1].strip())
        self.current_strategy = model[1].split('=')[1].strip()
        self.no_of_neighbors = int(model[2].split('=')[1].strip())
        self.train_x = np.loadtxt('trained_models/nearest/' + model[3].split('=')[1].strip())
        self.train_y = np.loadtxt('trained_models/nearest/' + model[4].split('=')[1].strip())

    def calculate_euclidean_distance(self, x, y):  # one vs all
        distances = -2 * np.matmul(x, y.T)
        distances += np.sum(x ** 2, axis=1)[:, np.newaxis]
        distances += np.sum(y ** 2, axis=1)
        return distances
