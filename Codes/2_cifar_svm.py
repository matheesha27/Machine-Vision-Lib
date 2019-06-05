import cv2
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# bag of words to identify clusters
N = 1000
sift_features = np.empty(shape=(0, 128))
for img in X_train[:N, :, :, :]:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    extractor = cv2.xfeatures2d.SIFT_create()
    kp, des = extractor.detectAndCompute(gray_img, None) # Key points and Descriptors
    if des is None:
        continue
    assert des.shape[1] == 128, 'wrong shape'
    sift_features = np.concatenate([sift_features, des], axis=0)

# kmeans clustering
k_means = KMeans(n_clusters=50, random_state=0).fit(sift_features)

# prepare train data
feature_vectors = np.zeros(shape=(N, 50))
for i, img in enumerate(X_train[:N, :, :, :]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    extractor = cv2.xfeatures2d.SIFT_create()
    kp, des = extractor.detectAndCompute(gray, None)
    if des is None:
        continue
    hist = k_means.predict(des)
    for val in hist:
        feature_vectors[i][val] += 1

# prepare test data
test_vectors = np.zeros(shape=(N, 50))
for i, img in enumerate(X_test[:N, :, :, :]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    extractor = cv2.xfeatures2d.SIFT_create()
    kp, des = extractor.detectAndCompute(gray, None)
    if des is None:
        continue
    hist = k_means.predict(des)
    for val in hist:
        test_vectors[i][val] += 1

# run experiment
labels = y_train[:N, 0]
test_labels = y_test[:N, 0]
model = svm.SVC(
                kernel='rbf',
                C=10,
                gamma=0.01
            )
model = model.fit(feature_vectors, labels)

train_accuracy = model.score(feature_vectors, labels)
test_accuracy = model.score(test_vectors, test_labels)
print("train_accuracy: {} \n test_accuracy: {}".format(train_accuracy, test_accuracy))