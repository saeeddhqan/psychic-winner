
# Python AI challenge

This repository contains a classification model implemented using PyTorch and sklearn. We utilized Scikit-learn, Pandas, and Matplotlib for numerical analysis, feature engineering, and more. Furthermore, we used Genetic Algorithm for Neural Architecture Search, hyperparameter tuning to optimize the model's performance, and feature selection. A modified version of [Population Based Training](https://www.deepmind.com/blog/population-based-training-of-neural-networks)(PBT) was implemented for training models. The final accuracy I got is <b>0.841</b> with Torch, but I got > 0.85 when I used oversampling technique. I used Pytest for testing, pyflakes as the linter, and pycodestyle for style.

## Files

- `model_sklearn.ipynb`: This notebook aims to provide a general understanding of the dataset by utilizing visualizations, numerical analysis, feature engineering and finally modeling.
- `model.py`: This file contains a manually designed neural network implemented using PyTorch, achieving 82% accuracy on the test set.
- `model_nas.ipynb`: This notebook presents a neural architecture search (NAS) utilizing the GA method to find the optimal neural network architecture for the churn dataset.
- `model_hype_opt.ipynb`: This notebook focuses on hyperparameter tuning using GA based on the NAS found elements for the network architecture.
- `model_optimized.py`: This is the optimized model, which is the result of NAS and hyperparameter space search.
- `feature_selection.ipynb`: This notebook focuses on feature selection using GA, utilizing crisp data for filtering features. I got 0.841 accuracy only because of this algorithm.
- `model_pbt.py`: This is an adjusted version of Population Based Training(PBT). I added some features to the base algorithm(such as elitism, usual cortex mutation, and early stopping). It was able to acheive 0.836 accuracy, outperforming the previous model(model_optimized.py with 0.831 accuracy)
- `model_optimized_pbt.py`: It runs the top model produced by PBT.
- `model_optimized_fs.py`: It runs the best performed model(0.841 accuracy) that filters features based on selected features.
- `util/util.py`: Utility functions.
- `util/elitism.py`: Using elitism technique for Genetic Algorithms. For further detail, look at the model_nas, where we describe the method.
- `util/torch_model_arch.py`: The architecture of torch models are in this file.
- `util/pbt_util.py`: PBT utilities, such as explore/exploit functions.
- `util/pbt_trainer.py`: Trainer class for PBT.

After tuning the neural network with `model_nas` and `model_hype_opt`, the accuracy increased by approximately ~1%. The PBT algorithm goes a little further(from 0.831 to 0.836). PBT is almost 2x faster than model_nas and model_hype_opt for producing top models. Finally, training a model by removing redundant features gave us ~1% more accuracy.

## colloquial description
I started creating a torch model, trained it on 90% of dataset, tested it on the rest, and in the first executions, I got >= 80% accuracy. After my first commit, I wanted to apply the Genetic Algorithm for Neural Architecture Search, hyperparameter space search, and feature selection. I had the elitism code before; encoding chromosomes, etc. weren't an issue, but, it took me forever to optimize generations, individuals, mutation_prob, crossover_prob parameters, and search the space with GA. And in this process, I made a blunder mistake: Doing feature selection after NAS and hyperparameter tuning. btw, I got what I wanted but I might get better results(who knows?). I did a lot of work to explore the space. The model_pbt works perfectly, especially when I implemented some of my own ideas. With sklearn, I got much better results when I used borderline SMOTE for the skewed class. I didn't experiment the effect of borderline SMOTE on Torch models. I hope I have met (at least some of) your expectations.

## Requirements

The following packages are required to run the notebooks and scripts in this repository:

- Torch
- Scikit-learn, catboost, xgboost, imblearn, and lightgbm
- Pandas
- Matplotlib, seaborn
- Deap
- Numpy
- tqdm
- pytest, pyflakes, and pycodestyle for test, linter, and style

## How to Use

To run the classification model, follow these steps:

    - Clone this repository.
    - Install the required packages listed in requirements.txt.
    - Make sure there's no error in project by running `pytest -v` command
    - Run files in the following order: 
        1. model_sklearn.ipynb
        2. model.py
        3. model_nas.ipynb
        4. model_hype_opt.ipynb
        5. feature_selection.ipynb
        6. model_optimized.py
        7. model_optimized_fs.py
        7. model_pbt.py
        8. model_optimized_pbt.py: Note that you need to change `model_path` variable depending on the top model.

Please note that running model_nas, model_hype_opt, feature_selection, and model_pbt is time-consuming depending on your system.
