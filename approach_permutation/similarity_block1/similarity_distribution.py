
import os
import numpy as np

def generate_plot(correct_rows, false_rows):
  
    # try:
    import matplotlib.pyplot as plt

    n_bins = 100

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.suptitle('Permutation Approach (block size 1)', fontsize=16)
    # We can set the number of bins with the *bins* keyword argument.
    correct_rows = np.array(correct_rows)
    false_rows = np.array(false_rows)
    print(correct_rows.shape)
    print(false_rows.shape)
    axs[0].hist(correct_rows[:, 1], bins=n_bins, color ='green')
    axs[1].hist(false_rows[:, 1], bins=n_bins, color ='red')
    axs[0].set_title('Memorized')
    axs[0].set(xlabel='Similarity Measure', ylabel='Number of Samples')
    axs[1].set_title('Not Memorized')
    axs[1].set(xlabel='Similarity Measure', ylabel='Number of Samples')

    plot_path = "./permute_distribution.png"
    plt.savefig(plot_path)
    print("A full error curve is located at " + str(plot_path))
        
    # except:
    #     print("Can't generate error curve; please install matplotlib to see the plot")


generation = np.load("./tmp/sample/generations/0.npy")
losses = np.load("./tmp/sample/losses/0.npy")
answers = np.load("../datasets/train_suffix.npy")[: -1000]

assert len(generation) == len(losses)

correct_rows = []
false_rows = []

for idx in range(len(generation)):
    # print(generation)
    guess = list(map(int, generation[idx]))
    guess = np.array(guess)

    guess_is_correct = np.all(answers[idx][-50:] == guess[-50:])
    loss = losses[idx]

    if guess_is_correct: 
        correct_rows.append([idx, loss])
    else:    
        false_rows.append([idx, loss])

generate_plot(correct_rows, false_rows)
