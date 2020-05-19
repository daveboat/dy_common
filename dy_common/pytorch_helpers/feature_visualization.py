"""
Visualzation tools for feature vectors using PCA and t-SNE

Built to be model-agnostic, as the functions here only require feature vectors and labels
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def feature_visualizer(features, labels, visualization_dim=2, method='pca', output='display',
                       output_file='feature_vis.png'):
    """
    Visualization function. Displays or saves a plot based on the chosen dimensionality reduction algorithm

    features: an iterable of feature arrays (numpy array or list)
    labels: an iterable of labels. Can be integer labels or strings. If integer labels, the labels are expected to
    start at 0 and end at the number of classes - 1. If string labels, will be automatically converted to integers
    using unique string labels in no particular order. These labels are for plotting purposes.
    visualization_dim: One of [2, 3]
    method: one of ['pca', 'tsne']
    output: one of ['display', 'save]
    output_file: if output=='save', figure will be saved to this file

    :return a matplotlib figure
    """

    # some assertion checks
    assert visualization_dim in [2, 3], 'visualization_dim ({}) must be 2 or 3'.format(visualization_dim)
    assert method in ['pca', 'tsne'], 'method ({}) must be \'pca\' or \'tsne\''
    assert output in ['display', 'save'], 'output_method ({}) must be \'display\' or \'save\''

    # so that we have the same behaviour for int and string labels, find the unique labels, then map them to integers
    # from 0 to num_classes
    unique_labels = list(set(labels))
    num_classes = len(unique_labels)
    print('Label order: {}'.format({unique_labels[i]: i for i in range(num_classes)}))
    int_labels = [unique_labels.index(l) for l in labels]

    # send features to dimensionality reduction algorithm
    if method == 'pca':
        result = PCA(n_components=visualization_dim).fit_transform(features)
    elif method == 'tsne':
        result = TSNE(n_components=visualization_dim).fit_transform(features)

    # plot
    if visualization_dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        scatter = ax.scatter(result[:, 0], result[:, 1], c=int_labels, cmap='hsv')
    elif visualization_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=int_labels, cmap='hsv')

    # add legend
    ax.legend(handles=scatter.legend_elements()[0], labels=unique_labels)

    # show or save plot
    if output == 'display':
        plt.show()
    elif output == 'save':
        plt.savefig(output_file, format='png')

    return fig


if __name__ == '__main__':
    np.random.seed(42)

    feature_dim = 512
    feature_count = 100
    num_class = 4

    features = np.random.randn(feature_count, feature_dim)
    labels = np.random.randint(0, num_class, (feature_count,))
    labels_string = np.empty(labels.shape, dtype=object)
    for i in range(len(labels)):
        if labels[i] == 0:
            labels_string[i] = 'aaaa'
        elif labels[i] == 1:
            labels_string[i] = 'bbbb'
        elif labels[i] == 2:
            labels_string[i] = 'cccc'
        elif labels[i] == 3:
            labels_string[i] = 'dddd'
    labels_string = list(labels_string)
    print(labels)
    #feature_visualizer(features, labels_string, visualization_dim=2, method='pca', output='display')

    feature_visualizer(features, labels, visualization_dim=3, method='tsne', output='display')