# This Python file uses the following encoding: gbk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm


def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)

    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    X += np.random.normal(scale=noise, size=X.shape)
    return X, y


def evaluate_classifier(classifier_func, X, y, X_test, y_test):
    accuracy = 0
    for _ in range(10):
        accuracy += classifier_func(X, y, X_test, y_test)
    return accuracy / 10


def decision_tree_classifier(X, y, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return accuracy_score(y_test, clf.predict(X_test))


def adaboost_classifier(X, y, X_test, y_test):
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200,
        learning_rate=1,
        algorithm='SAMME',
        random_state=42)
    ada_clf.fit(X, y)
    return accuracy_score(y_test, ada_clf.predict(X_test))


def svm_classifier(X, y, X_test, y_test, kernel='rbf'):
    clf = svm.SVC(kernel=kernel, C=1.0, random_state=42)
    clf.fit(X, y)
    return accuracy_score(y_test, clf.predict(X_test))


if __name__ == "__main__":
    # Generate data
    X, y = make_moons_3d(n_samples=1000, noise=0.2)
    X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)

    # Evaluate classifiers
    classifiers = [
        ("决策树分类器", lambda X, y, Xt, yt: decision_tree_classifier(X, y, Xt, yt)),
        ("AdaBoost+决策树分类器", lambda X, y, Xt, yt: adaboost_classifier(X, y, Xt, yt)),
        ("SVM(线性核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'linear')),
        ("SVM(RBF核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'rbf')),
        ("SVM(多项式核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'poly')),
        ("SVM(sigmoid核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'sigmoid'))
    ]

    for name, clf in classifiers:
        accuracy = evaluate_classifier(clf, X, y, X_test, y_test)
        print(f"{name}的准确率: {accuracy:.4f}")

    # Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot training data
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    # Create grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50),
                             np.linspace(z_min, z_max, 50))

    # Predict and plot decision boundary
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    clf = svm.SVC(kernel='rbf', C=1.0, random_state=42)
    clf.fit(X, y)
    grid_predictions = clf.predict(grid_points).reshape(xx.shape)
    ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=grid_predictions, cmap='viridis', marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Make Moons with SVM Classification')
    plt.show()