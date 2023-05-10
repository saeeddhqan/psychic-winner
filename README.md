# Python AI challenge

This repository contains a classification model implemented using PyTorch. We also utilized Scikit-learn, Pandas, and Matplotlib for numerical analysis, feature engineering, and more. Furthermore, we used Genetic Algorithm for Neural Architecture Search and hyperparameter tuning to optimize the model's performance. The final commit was for  [Population Based Training](https://www.deepmind.com/blog/population-based-training-of-neural-networks)(PBT); an approach for training and adjusting hyperparameters.

## Files

- `model_sklearn.ipynb`: This notebook aims to provide a general understanding of the dataset by utilizing visualizations, numerical analysis, and feature engineering.
- `model.py`: This file contains a manually designed neural network implemented using PyTorch, achieving 82% accuracy on the test set.
- `model_nas.ipynb`: This notebook presents a neural architecture search (NAS) utilizing the GA method to find the optimal neural network architecture for the churn dataset.
- `model_hype_opt.ipynb`: This notebook focuses on hyperparameter tuning using GA based on the NAS found elements for the network architecture.
- `model_optimized.py`: This is the optimized model, which is the result of NAS and hyperparameter space search.
- `feature_selection.ipynb`: This notebook focuses on feature selection using GA, utilizing crisp data for filtering features.
- `model_pbt.py`: This is an adjusted version of Population Based Training(PBT).  I added some features to the base algorithm(such as elitism, usual cortex mutation, and early stopping). It was able to acheive 0.836 accuracy, outperforming the previous model(model_optimized.py with 0.831 accuracy)
- `model_optimized_pbt.py`: It runs the top model trained by PBT.
- `util/util.py`: Utility functions.
- `util/elitism.py`: Using elitism technique in Genetic Algorithm. For further detail, look at the model_nas, where we describe the method.
- `util/torch_model_arch.py`: The architecture of torch models are in this file.
- `util/pbt_util.py`: PBT utilities, such as explore/exploit functions.
- `util/pbt_trainer.py`: Trainer class for PBT.

After tuning the neural network with `model_nas` and `model_hype_opt`, the accuracy increased by approximately ~1%. So, the final accuracy is 83%.

## colloquial description
I started creating a torch model, trained it on 90% of dataset, tested it on the rest, and in the first executions, I got >= 80% accuracy. I asked a question in the group about whether it is a must to use sklearn for modeling or not; and that's why I started using torch. I created this repo in private mode but I needed to invite mason-chase as a contributor to invite him for reviewing the pr. After my first commit, I wanted to apply the Genetic Algorithm for Neural Architecture Search, hyperparameter space search, and feature selection. I had the elitism code before, and encoding chromosomes, etc. weren't an issue; But it took me forever to optimize generations, individuals, mutation_prob, crossover_prob parameters, and search the space with GA. And in this process, I made a blunder mistake: Doing feature selection after NAS and hyperparameter tuning. btw, I got what I wanted but I might get better results(who knows?). After all these things, I created model_sklearn for getting more insight of dataset and modeling with sklearn. I couldn't afford to invest the time on TDD, DDD, and BDD. But, if it is a must(which is the case), I need more time for TDD phase. I know I should've done this before doing anything. More detail about techniques can be found in notebooks. I hope I have met (at least some of) your expectations.

## Requirements

The following packages are required to run the notebooks and scripts in this repository:

- Torch
- Scikit-learn
- Pandas
- Matplotlib, seaborn
- Deap
- Numpy
- tqdm

## How to Use

To run the classification model, follow these steps:

    - Clone this repository.
    - Install the required packages listed in requirements.txt.
    - Run the notebooks in the following order: 
        1. model_sklearn.ipynb
        2. model.py
        3. model_nas.ipynb
        4. model_hype_opt.ipynb
        5. model_optimized.py
        6. feature_selection.ipynb
        7. model_pbt.py: If you try to run it twice without deleting individuals directory, it returns an error. So delete it.
        8. model_optimized_pbt.py: Note that you need to change `model_path` variable depending on the top model.

Please note that running model_nas, model_hype_opt, and feature_selection is time-consuming depending on your system.

