import random

import matplotlib.pyplot as plt
import numpy as np

def rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def sample_tr_2d(sample_num=100, x_range=(-10, 10), y_range=(-10, 10)):
    x_translations = np.random.uniform(x_range[0], x_range[1], sample_num)
    y_translations = np.random.uniform(y_range[0], y_range[1], sample_num)
    translations = np.column_stack((x_translations, y_translations))
    return translations


def sample_rot_2d(sample_num=100):
    thetas = np.random.uniform(0, 2*np.pi, sample_num)
    matrices = [rotation_matrix(theta) for theta in thetas]
    return matrices


def visual_2d(graphs):
    plt.figure(figsize=(8, 8))
    if isinstance(graphs, np.ndarray):
        graphs = graphs.tolist()

    if isinstance(graphs[0], tuple):
        graphs.append(graphs[0])
        x, y = [d[0] for d in graphs], [d[1] for d in graphs]
        plt.plot(x, y, '-')
    else:
        for graph in graphs:
            graph.append(graph[0])
            x, y = [d[0] for d in graph], [d[1] for d in graph]
            plt.plot(x, y, '-')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D point Cloud Visualizatin')
    plt.grid(True)
    plt.axis('equal')
    # plt.show()
    plt.savefig('data/test.png')

if __name__=='__main__':
    graph = [(1,1), (1,2), (2,2), (2,1)]
    visual_2d(graphs=graph)
