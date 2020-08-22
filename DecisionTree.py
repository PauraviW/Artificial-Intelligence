import numpy as np
import pickle


class Tree:
	def __init__(self):
		self.left = {}
		self.right = {}
		self.feature = None
		self.feature_value = None
		self.isTerminal = False
		self.label = 0
		self.feature_count = 0


class DecisionTree:
	def __init__(self, max_depth=5, minimum_num_leaves=10):
		self.max_depth = max_depth
		self.minimum_num_leaves = minimum_num_leaves
		self.root = None

	def save(self, fname):
		model_variables = locals()['self']
		f = open('trained_models/tree/' + fname, 'w')
		f.write('max_depth=%d\n' % self.max_depth)
		f.write('minimum_num_leaves=%d\n' % self.minimum_num_leaves)
		f.write('decision_tree_file=dt\n')
		with open('trained_models/tree/dt', 'wb') as tree:
			pickle.dump(model_variables.root, tree)
		f.close()

	def load(self, fname):
		f = open('trained_models/tree/' + fname, 'r')
		model = f.readlines()
		f.close()
		self.max_depth = int(model[0].strip().split('=')[1])
		self.minimum_num_leaves = int(model[1].strip().split('=')[1])
		with open('trained_models/tree/dt', 'rb') as tree:
			self.root = pickle.load(tree)

	def fit(self, x, y):
		self.root = self.build_decision_tree(np.append(x, y, axis=1), self.max_depth, self.minimum_num_leaves)

	def score(self, x, y):
		y_pred = self.predict(x)
		return np.count_nonzero(y_pred == y) / len(y_pred)

	def predict(self, x):
		y_pred = []
		for row in x:
			y_pred.append([self.get_prediction(row, self.root)])
		return np.array(y_pred)

	def get_prediction(self, row, tree):
		if tree.isTerminal:
			return tree.label
		if row[tree.feature] >= tree.feature_value:
			return self.get_prediction(row, tree.right)
		else:
			return self.get_prediction(row, tree.left)

	def calculate_info_gain(self, left, right, current_uncertainty):
		p = float(len(left)) / (len(left) + len(right))
		return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

	def data_per_label(self, data):
		count_array = np.asarray(np.unique(data[:, -1], return_counts=True))
		return dict(enumerate(zip(count_array[0], count_array[1])))

	def gini(self, data):
		data_per_label = self.data_per_label(data)
		impurity = 1
		for label in data_per_label:
			probability_of_label = float(data_per_label[label][1]) / len(data)
			impurity -= probability_of_label ** 2
		return impurity

	def get_best_features(self, data):
		best_gain = 0
		best_feature = None
		best_value = 0
		current_uncertainty = self.gini(data)
		number_of_features = len(data[0]) - 1

		for feature in range(number_of_features):
			feature_unique_values = np.unique(data[:, feature])
			feature_unique_values = [np.sum(feature_unique_values) / len(feature_unique_values)]
			for value in feature_unique_values:
				right_tree, left_tree = self.split_tree(data, feature, value)
				if len(right_tree) == 0 or len(left_tree) == 0:
					continue
				gain = self.calculate_info_gain(right_tree, left_tree, current_uncertainty)

				if gain >= best_gain:
					best_gain, best_feature, best_value = gain, feature, value

		return best_gain, best_feature, best_value

	def build_decision_tree(self, data, max_depth, minimum_num_leaves):
		root = Tree()
		info_gain, feature, value = self.get_best_features(data)

		if info_gain == 0 or max_depth <= 0 or minimum_num_leaves <= root.feature_count:
			root.isTerminal = True
			root.label = self.most_frequent_label(data)
			return root

		root.feature = feature
		root.feature_value = value
		root.feature_count += 1
		right, left = self.split_tree(data, feature, value)

		right = self.build_decision_tree(right, max_depth - 1, minimum_num_leaves)
		left = self.build_decision_tree(left, max_depth - 1, minimum_num_leaves)

		root.left = left
		root.right = right

		return root

	def split_tree(self, data, feature, value):
		left, right = [], []
		for row in data:
			if row[feature] >= value:
				right.append(row)
			else:
				left.append(row)
		return np.array(right), np.array(left)

	def most_frequent_label(self, data):
		(values, counts) = np.unique(data[:, -1], return_counts=True)
		return values[np.argmax(counts)]
