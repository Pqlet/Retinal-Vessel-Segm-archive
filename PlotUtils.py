import matplotlib.pyplot as plt
import numpy as np

def plot_history(train_history, val_history, title='loss', save_folder=None):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)

    steps = list(range(0, len(train_history), int(len(train_history) / len(val_history))))

    plt.plot(steps, val_history, c='orange', label='val')
    plt.xlabel('train steps')

    plt.legend(loc='best')
    plt.grid()

    if save_folder is not None:
        plt.savefig(f'{save_folder}/{title}.png')
    else:
        plt.show()