import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    avg = np.zeros(len(scores))
    for i in range(len(avg)):
        avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, avg)
    plt.title('Average of previous 100 scores')
    plt.savefig(figure_file)
