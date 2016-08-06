#!/usr/bin/python

from tensorflow.contrib import learn
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = learn.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
