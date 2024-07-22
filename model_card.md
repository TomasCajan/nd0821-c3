# Model Card

    ---

    ## Model Details
    This is a Logistic Regression model trained on the census dataset.
    It was trained using a k-fold cross-validated grid search and comes with the following setup:  LogisticRegression(C=0.01, max_iter=500)

    ---

    ## Intended Use
    Created for classification of salary categories within the census dataset.

    ---

    ## Training Data
    Model was trained on the census+income dataset publicly available here:
    https://archive.ics.uci.edu/dataset/20/census+income
    Specifically on 80% of the dataset, having exact shape of (26048, 15)
    

    ---

    ## Evaluation Data
    Model was evaluated on the remaining 20% of the dataset mentioned above, specifically : (6513, 15)
    
    ---

    ## Metrics
    Model scores following metrics of:
    - Precision: 0.7115
    - Recall: 0.2764
    - F1 Score: 0.3982

    ---

    ## Ethical Considerations
    Model unfortunately has tendency to positively discriminate white and Asian people.

    ---

    ## Caveats and Recommendations
    Use at your own risk.