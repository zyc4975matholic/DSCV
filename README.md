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
* The code for plotting are in plot_code folder
* The code for experiments are in code folder

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
  + classifier: A classifier object( Not instance) that at least implement fit and predict methods, default = SVC
  + kwargs: A dictionary that contains the parameter-value pairs used to generate the classifier, default = default setting of SVC
  + sample_rate: A float between 0 and 1, indicates the test sample size, default = 0.2
  + k_fold: A int larger than 2, indicates the number of folds in cross validation, default = 3
  + resampler: A str that indicates the sampling strategy. Must be one of the following, default = "SMOTE"
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
