# MFMO-model

feature extraction:  run MFMO/filterfeature_batch_extract.py to extract multi-filter features in batches


MFMO : 
  
    NMI-NSGAII : run the MFMO/NMI-NSGAII/main.m to select features by the multi-objectives-based feature selection algorithm
  
    MIC-mRMR : run the MFMO/MICmRMR_sel.py to restrict the number of selected feature and get top 10 important features from each filter
  
    Identification of radiomic biomarkers : run the MFMO/sel_shap.py to get the final 10 radiomic biomarkers
    
    comparision with previous works
    
    comparision LR classifier with other classifier


Data:

    train_biomarker.xlsx : 10 signature features extracted from training dataset
    
    test_BraTS2013_biomarker.xlsx : 10 signature features extracted from BraTS 2013 test dataset
