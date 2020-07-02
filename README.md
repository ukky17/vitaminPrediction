# Vitamin deficiency predicted by machine-learning

This repository contains codes used to produce most of the machine-learning results of the following paper.

* **Efficient Prediction of Vitamin B Deficiencies via Machine-learning Using Routine Blood Test Results in Patients With Intense Psychiatric Episode**. Hidetaka Tamune\*, Jumpei Ukita\*, Yu Hamamoto, Hiroko Tanaka, Kenji Narushima and Naoki Yamamoto. *Frontiers in Psychiatry*, 2020. [Paper Link](https://www.frontiersin.org/articles/10.3389/fpsyt.2019.01029/abstract). (\*: co-first author)

## Usage
1. Modify `load_dataset` in `data_utils.py` to read the dataset.
2. Optimize hyperparameters by running `grid_search.py`.
3. Calculate the prediction accuracy: e.g. `python test.py --target folate --modelType rf`
4. To estimate the sample size necessary for a high prediction accuracy, run `sample_size.py`
