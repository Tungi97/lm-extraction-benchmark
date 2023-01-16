
import os
import numpy as np

def generate_plot(correct_rows, false_rows):
  
    try:
        import matplotlib.pyplot as plt

        n_bins = 100
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        # We can set the number of bins with the *bins* keyword argument.
        axs[0].hist(correct_rows[:,1], bins=n_bins, density = 1, color ='green')
        axs[1].hist(false_rows[:,1], bins=n_bins, density = 1, color ='red')
        
        plt.title('Permutation Approach')

        axs[0].xlabel("Similarity")
        axs[0].ylabel("Number of Samples")
        axs[0].set_title('Memorized')
        axs[1].xlabel("Similarity")
        axs[1].ylabel("Number of Samples")
        axs[1].set_title('Not Memorized')
        
        plot_path = "./permute_similarity_distribution.png"
        plt.savefig(plot_path)
        print("A full distribution histogram is located at " + str(plot_path))
        
    except:
        print("Can't generate distribution histogram; please install matplotlib to see the plot")


generation = np.load("./tmp/sample/generations/0.npy")
losses = np.load("./tmp/sample/losses/0.npy")
answers = np.load("../datasets/train_suffix.npy")[:-1000]

assert len(generation) == len(losses)

correct_rows = []
false_rows = []

for idx in range(len(generation)):
    
    guess = list(map(int,generation[idx]))
    guess = np.array(guess)

    guess_is_correct = np.all(answers[idx][-50:] == guess[-50:])
    loss = losses[idx]

    if guess_is_correct: 
        correct_rows.append([idx, loss])
    else:    
        false_rows.append([idx, loss])

correct_rows = np.array(correct_rows)
false_rows = np.array(false_rows)

print(len(false_rows))
print(len(correct_rows))

generate_plot(correct_rows, false_rows)
