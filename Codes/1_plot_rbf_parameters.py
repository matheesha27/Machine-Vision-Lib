'''
==================
RBF SVM parameters
==================

This example illustrates the effect of the parameters ``gamma`` and ``C`` of
the Radial Basis Function (RBF) kernel SVM.

Intuitively, the ``gamma`` parameter defines how far the influence of a single
training example reaches, with low values meaning 'far' and high values meaning
'close'. The ``gamma`` parameters can be seen as the inverse of the radius of
influence of samples selected by the model as support vectors.

The ``C`` parameter trades off correct classification of training examples
against maximization of the decision function's margin. For larger values of
``C``, a smaller margin will be accepted if the decision function is better at
classifying all training points correctly. A lower ``C`` will encourage a
larger margin, therefore a simpler decision function, at the cost of training
accuracy. In other words``C`` behaves as a regularization parameter in the
SVM.

The first plot is a visualization of the decision function for a variety of
parameter values on a simplified classification problem involving only 2 input
features and 2 possible target classes (binary classification). Note that this
kind of plot is not possible to do for problems with more features or target
classes.

The second plot is a heatmap of the classifier's cross-validation accuracy as a
function of ``C`` and ``gamma``. For this example we explore a relatively large
grid for illustration purposes. In practice, a logarithmic grid from
:math:`10^{-3}` to :math:`10^3` is usually sufficient. If the best parameters
lie on the boundaries of the grid, it can be extended in that direction in a
subsequent search.

Note that the heat map plot has a special colorbar with a midpoint value close
to the score values of the best performing models so as to make it easy to tell
them apart in the blink of an eye.

The behavior of the model is very sensitive to the ``gamma`` parameter. If
``gamma`` is too large, the radius of the area of influence of the support
vectors only includes the support vector itself and no amount of
regularization with ``C`` will be able to prevent overfitting.

When ``gamma`` is very small, the model is too constrained and cannot capture
the complexity or "shape" of the data. The region of influence of any selected
support vector would include the whole training set. The resulting model will
behave similarly to a linear model with a set of hyperplanes that separate the
centers of high density of any pair of two classes.

For intermediate values, we can see on the second plot that good models can
be found on a diagonal of ``C`` and ``gamma``. Smooth models (lower ``gamma``
values) can be made more complex by increasing the importance of classifying
each point correctly (larger ``C`` values) hence the diagonal of good
performing models.

Finally one can also observe that for some intermediate values of ``gamma`` we
get equally performing models when ``C`` becomes very large: it is not
necessary to regularize by enforcing a larger margin. The radius of the RBF
kernel alone acts as a good structural regularizer. In practice though it
might still be interesting to simplify the decision function with a lower
value of ``C`` so as to favor models that use less memory and that are faster
to predict.

We should also note that small differences in scores results from the random
splits of the cross-validation procedure. Those spurious variations can be
smoothed out by increasing the number of CV iterations ``n_splits`` at the
expense of compute time. Increasing the value number of ``C_range`` and
``gamma_range`` steps will increase the resolution of the hyper-parameter heat
map.

'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# Dataset for grid search
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)


# Training and getting the best parameters
C_range = np.array([0.1, 1, 10, 100, 1000])  # range for parameter C
gamma_range = np.array([0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1])  # range for parameter Gamma

param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(test_size=0.2, random_state=42)
grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)
print("The best parameters from Training are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

# Validating on test data using the best parameters
print('Training Accuracy = ', grid.score(X_train, y_train))
#print('Test Accuracy = ', grid.score(X_test, y_test))
print('Test Accuracy from scratch 0.1 = ', SVC(C=10, kernel='rbf', gamma=0.1).fit(X_train, y_train).score(X_test, y_test))

train_scores = np.zeros((5,7))
test_scores = np.zeros((5,7))
for c in range(len(C_range)):
    for gamma in range(len(gamma_range)):
        svc = SVC(C=C_range[c], gamma=gamma_range[gamma], kernel='rbf')
        svc.fit(X_train, y_train)
        train_scores[c, gamma] = svc.score(X_train, y_train)
        test_scores[c, gamma] = svc.score(X_test, y_test)


print('Train Scores: \n', train_scores)
print('Test Scores: \n', test_scores)

'''
train_scores2 = grid.cv_results_['mean_train_score'].reshape(len(C_range), len(gamma_range))
print('Train Scores2: \n', train_scores2)
test_scores2 = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
print('Test Scores2: \n', test_scores2)
'''

# Heatmap for Training Accuracy
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(train_scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.732))
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
           norm=MidpointNormalize(vmin=0.2, midpoint=0.732))
plt.xlabel('Gamma')
plt.ylabel('SVM Regularization Parameter (C)')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Test Accuracy')
plt.show()