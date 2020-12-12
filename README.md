# DSCV: Double Sampling Cross Validation Framework

By the co-work of Yuchen Zhu and Siyu Gao

This repo contains all the code implementation and experiment details for the project DSCV: Unknown Unknown Detection in Imbalance Learning, which is an algorithmic framework that can address both imbalanced learning and open set recognition at the same time.

User Guide
-
### Repo Structure
* Our research with full details can be found in the paper (the pdf file)
* To use our proposed framework DSCV, use DSCV.py
* The real-world datasets we use (LETTER, PENDIGITS, COIL20) are in Real-World dataset folder
* The synthetic datsets we generate are in Synthetic dataset folder
* Some utils are contained in utils folder
* The code for plotting are in plot_code folder
  + Separability_Graph.ipynb plots the visualization of well/ill separated data
  + real_dataset_Graph.ipynb plots the experiment results on LETTER,COIL20 and PENDIGITS
  + synthetic_dataset_Graph.ipynb plots the experiment results on Synthetic datasets
  + Toy_Example_Graph.ipynb plots the visualization of the algorithm applied to a simple dataset
* The code for experiments are in code folder
  + RTSCV.py contains the majority of the code written during the research (not well commented and arranged! upload for archive)
  + OSR_PENDIGITS.py contains the code to run WSVM, 1-vs-set Machine on PENDGIITS
  + OSR_LETTER.py contains the code to run WSVM, 1-vs-set Machine on LETTER
  + OSR_COIL20_part1.py/OSR_COIL20_part2.py contains the code to run WSVM, 1-vs-set Machine on COIL20
  + generate_LT.py contains the code that generates Long-tailed dataset from real world dataset
  + plot_toy_example.py contains the code that generates the data as the toy example visualization of the framework
  + sampling.py contains the code that run the experiments on the combination of DSCV with different sampling methods
  + ensemble.py contains the code that runs the experiments on the combination of RTSCV with different ensemble classifier implemented with [Imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn/tree/master/imblearn)
  + more_ensemble.py contains the code that runs the experiments on the combination of RTSCV with different ensemble classifier implemented with [Awesome-imbalanced-learning](https://github.com/ZhiningLiu1998/awesome-imbalanced-learning)
### To-start
* All code are written in Python.
* Download the repo by
  ```
  git clone git@github.com:zyc4975matholic/DSCV.git
  cd DSCV
  ```
* Install all dependencies by
  ```
  pip install -r requirements.txt
  ```
* Test the code by
  ```
  python DSCV.py
  ```

### Initialize a DSCV instance
* To intialize a DSCV instance the following parameter has to be passed
  + classifier: a classifier object(not instance) that at least implement fit and predict methods, default = SVC
  + kwargs: a dictionary that contains the parameter-value pairs used to generate the classifier, default = default setting of SVC
  + sample_rate: a float between 0 and 1, indicates the test sample size, default = 0.2
  + k_fold: a int larger than 2, indicates the number of folds in cross validation, default = 3
  + resampler: a str that indicates the sampling strategy. Must be one of the following, default = "SMOTE"
      - "ClusterCentroids"
      - "CondensedNearestNeighbour"
      - "EditedNearestNeighbours"
      - "RepeatedEditedNearestNeighbours"
      - "AllKNN" 
      - "InstanceHardnessThreshold"
      - "NearMiss"
      - "NeighbourhoodCleaningRule"
      - "OneSidedSelection"
      - "RandomUnderSampler"
      - "TomekLinks"
      - "SVM-SMOTE"
      - "ADASYN"
      - "KMeansSMOTE"
      - "BorderlineSMOTE"
      - "RandomOverSampler"
      - "SMOTE"
      - "SMOTEENN"
      - "SMOTETomek"

### Trainning with DSCV
* use method meta_fit, passing in X_train, Y_train, X_test, Y_test
  return a trained model with DSCV

### Predict with DSCV
* use method predict, passing X,
  return a numpy array with predicted labels, label with 99 are unknown unknowns found by DSCV
  
  
Citation
-
If you found our work useful to your research, please cite

```
@misc{DSCV2020,
  author = {Yuchen, Zhu and Siyu, Gao},
  title = {DSCV},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zyc4975matholic/DSCV}}
}
```

Acknowledgement
-
This project is finished during Fall 2020 Machine Learning course. We thanks Prof. Enric Junqué de Fortuny and TA Zijian Zhou for their meaningful advices on the project. The project also can't be done without the inspiration from [Supervised Discovery of Unknown Unknowns through Test Sample Mining](https://ojs.aaai.org//index.php/AAAI/article/view/7252) and [Rethinking the Value of Labels for Improving Class-Imbalanced Learning](https://arxiv.org/abs/2006.07529). 
