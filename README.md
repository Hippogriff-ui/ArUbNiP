# UbNiRF
Prediction of protein ubiquitination sites in Arabidopsis thaliana and Homo sapiens

#UbNiRF uses the following programming languages and versions:
* python 3.10
* python 3.6
* MATLAB 2016a


#Guiding principles:

**The dataset folder contains ubiquitination site datasets for Arabidopsis thaliana and Homo sapiens.\n
**The code folder is the code implementation in the article.\n
**The "AA531properties.xlsx" file contains 531 physical and chemical properties of amino acids.

**feature extraction:
   BE.py is the implementation of BE.\n
   CKSAAP.py is the implementation of CKSAAP.\n
   EAAC.py is the implementation of EAAC.\n
   PWM.py is the implementation of PWM.\n
   AA531.py is the implementation of AA531.\n
   "PSSM" folder is the implementation of PSSM.
   
**feature selection:
   Elastic_net.py represents the Elastic net.\n
   MRMR.py represents the mRMR.\n
   Null_importances.py represents the Null importances.
  
**classifiers:
   DT_Kfold_para.py and DT_Kfold_Test.py are the implementation of DT.\n
   RF_Kfold_para.py and RF_Kfold_Test.py are the implementation of RF.\n
   XGBoost_Kfold_para.py and XGBoost_Kfold_Test.py are the implementation of XGBoost.\n
   SVC_Kfold_para.py and SVC_Kfold_Test.py are the implementation of SVM.

   **The "chi2_window.py" file is the implementation of the chi-square test with graphing.
   
   **The "feature_combine.py" is the implementation of combination of 6 features.
   
   **The "ROC_curve.py" is the implementation of ROC curve plotting for different feature selection methods and classifier combinations.
   
   **The "train_feature2test.py" file is to apply the features selected for the training set to the test set.

   
