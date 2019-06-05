import numpy as np
import cvxopt
import cvxopt.solvers
from numpy import linalg
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler


def linear_kernel(x1, x2, _=None):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def rbf_kernel(x, y, gamma=0.001):
    return np.exp((-linalg.norm(x - y) ** 2) * gamma)


KERNEL_DICT = {
    "linear": linear_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel
}


class SVM(object):

    def __init__(self, kernel='rbf', c=None, param=None, verbose=False):
        """
        Initialize SVM Object
        :param kernel:          str (linear / poly / rbf)
        :param c:               regularization term
        :param verbose:         print to console
        :param param:           gamma / polynomial degree parameter
        """
        self.kernel = KERNEL_DICT[kernel]
        self.C = c
        self.param = param
        self.verbose = verbose
        if not verbose:
            cvxopt.solvers.options['show_progress'] = False

        self.a = None
        self.sv = None
        self.sv_y = None
        self.w = None
        self.b = None

    def fit(self, x, y):
        """
        Fitting function
        :param x:       array of shape (num_examples, num_dims)
        :param y:       array of shape (num_examples, )
        :return:        No return
        """
        n_samples, n_features = x.shape

        # Gram matrix
        k_mat = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                k_mat[i, j] = self.kernel(x[i], x[j], self.param)

        _P = cvxopt.matrix(np.outer(y, y) * k_mat)
        _q = cvxopt.matrix(np.ones(n_samples) * -1)
        _A = cvxopt.matrix(y, (1, n_samples), tc='d')
        _b = cvxopt.matrix(0.0)

        if self.C is None:
            _G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            _h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = - np.identity(n_samples)
            tmp2 = np.identity(n_samples)
            _G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            _h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(_P, _q, _G, _h, _A, _b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = x[sv]
        self.sv_y = y[sv]
        if self.verbose:
            print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * k_mat[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, x):
        """
        Helper for predict. Same input as predict.
        :param x:       array of shape (num_examples, num_dims)
        :return:        array of shape (num_examples, )
        """
        if self.w is not None:
            return np.dot(x, self.w) + self.b
        else:
            y_predict = np.zeros(len(x))
            for i in range(len(x)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(x[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, x):
        """
        Predict function
        :param x:       array of shape (num_examples, num_dims)
        :return:        array of shape (num_examples, )
        """
        return np.sign(self.project(x))


class MultiClassSVM:

    def __init__(self, kernel='rbf', c=None, param=None, verbose=False):
        """
        Initialize SVM Object
        :param kernel:          str (linear / poly / rbf)
        :param c:               regularization term
        :param param:           gamma / polynomial degree parameter
        """
        self.kernel = kernel
        self.C = c
        self.param = param
        self.verbose = verbose
        self.classifiers = []
        self.classes = None

    def fit(self, x, y):
        """

        :param x:       array of shape (num_examples, num_dims)
        :param y:       array of shape (num_examples, )
        :return:        No return
        """
        self.classes = np.unique(y)
        for class_id in self.classes:
            classifier = SVM(kernel=self.kernel, c=self.C, param=self.param, verbose=self.verbose)
            y_ = ((y == class_id).astype(float) - 0.5) * 2
            classifier.fit(x, y_)
            self.classifiers.append(classifier)

    def predict(self, x):
        """
        Predict function
        :param x:       array of shape (num_examples, num_dims)
        :return:        array of shape (num_examples, )
        """
        assert len(self.classifiers) > 0, "run fit before prediction"
        scores = np.empty(shape=(x.shape[0], len(self.classifiers)))  # num_examples, num_classes
        for i, classifier in enumerate(self.classifiers):
            score = classifier.project(x)
            scores[:, i] = score
        return np.argmax(scores, axis=1)


iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

C_range = np.array([0.1, 1, 10, 100, 1000])  # range for parameter C
gamma_range = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1])  # range for parameter Gamma

train_scores = np.zeros((5,7))
test_scores = np.zeros((5,7))
for c in range(len(C_range)):
    for gamma in range(len(gamma_range)):
        clf = MultiClassSVM(c=C_range[c], param=gamma_range[gamma])
        clf.fit(x_train, y_train)
        train_pred = clf.predict(x_train)
        test_pred = clf.predict(x_test)

        train_scores[c, gamma] = accuracy_score(y_true=y_train, y_pred=train_pred)
        test_scores[c, gamma] = accuracy_score(y_true=y_test, y_pred=test_pred)


print('Train Scores: \n', train_scores)
print('Test Scores: \n', test_scores)

# HEATMAPS
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# Heatmap for Training Accuracy
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(train_scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.5))
plt.xlabel('Gamma')
plt.ylabel('SVM Regularization Parameter (C)')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Training Accuracy')
plt.show()

# Heatmap for Test Accuracy
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(test_scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.5))
plt.xlabel('Gamma')
plt.ylabel('SVM Regularization Parameter (C)')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Test Accuracy')
plt.show()

'''
clf = MultiClassSVM(c=10, param=0.1)
clf.fit(x_train, y_train)
train_pred = clf.predict(x_train)
test_pred = clf.predict(x_test)

train_acc = accuracy_score(y_true=y_train, y_pred=train_pred)
test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)

print("train_accuracy: {} \n test_accuracy: {}".format(train_acc, test_acc))
'''