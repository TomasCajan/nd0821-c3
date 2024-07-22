# Model Card

    ---

    ## Model Details
    This is a Logistic Regression model trained on the census dataset. It was trained using a k-fold cross-validated grid search and comes with the following setup:  LogisticRegression(C=1, max_iter=500, solver='liblinear')

    ---

    ## Intended Use
    Created for classification of salary categories within the census dataset.

    ---

    ## Training Data
    Model was trained on the following training dataset:  
    <class 'pandas.core.frame.DataFrame'>
Index: 26048 entries, 21237 to 21460
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             26048 non-null  int64 
 1   workclass       26048 non-null  object
 2   fnlgt           26048 non-null  int64 
 3   education       26048 non-null  object
 4   education-num   26048 non-null  int64 
 5   marital-status  26048 non-null  object
 6   occupation      26048 non-null  object
 7   relationship    26048 non-null  object
 8   race            26048 non-null  object
 9   sex             26048 non-null  object
 10  capital-gain    26048 non-null  int64 
 11  capital-loss    26048 non-null  int64 
 12  hours-per-week  26048 non-null  int64 
 13  native-country  26048 non-null  object
 14  salary          26048 non-null  object
dtypes: int64(6), object(9)
memory usage: 3.2+ MB


    ---

    ## Evaluation Data
    Model was evaluated on the following test dataset kept away from the train data:  
    <class 'pandas.core.frame.DataFrame'>
Index: 6513 entries, 21959 to 1274
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   age             6513 non-null   int64 
 1   workclass       6513 non-null   object
 2   fnlgt           6513 non-null   int64 
 3   education       6513 non-null   object
 4   education-num   6513 non-null   int64 
 5   marital-status  6513 non-null   object
 6   occupation      6513 non-null   object
 7   relationship    6513 non-null   object
 8   race            6513 non-null   object
 9   sex             6513 non-null   object
 10  capital-gain    6513 non-null   int64 
 11  capital-loss    6513 non-null   int64 
 12  hours-per-week  6513 non-null   int64 
 13  native-country  6513 non-null   object
 14  salary          6513 non-null   object
dtypes: int64(6), object(9)
memory usage: 814.1+ KB


    ---

    ## Metrics
    Model scores following metrics of:
    - Precision: 0.7000
    - Recall: 0.2586
    - F1 Score: 0.3776

    ---

    ## Ethical Considerations
    Model has tendency to positively discriminate white and Asian people.

    ---

    ## Caveats and Recommendations
    Use at your own risk.