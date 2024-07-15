## Credit Card Fraud Detection - EDA

**Author**
Polsani, Prashanth Kumar (https://github.com/polsanip)

### Executive Summary

For the credit card transaction prediction use case, I utilized a simulation dataset sourced from Kaggle, with fraud transactions accounting for 0.5% of the dataset.

Following the CRISP-DM methodology, I meticulously cleaned the data and performed feature engineering to prepare for model training. Further details on data processing and feature engineering can be found in the respective sections.

The models experimented with include:

* LogisticRegression
* KNN
* DecisionTreeClassifier
* SVM
* GaussianNB
* SGDClassifier

After training these models, I evaluated their performance using metrics such as f1_score and balanced score from the sklearn metrics library, focusing particularly on true positives.

Unfortunately, all models showed poor performance in predicting true positives. However, KNN and DecisionTreeClassifier demonstrated relatively better performance in this regard. For detailed outcomes, please refer to the results section.

Moving forward, as outlined in the next steps of this README, it is crucial to further explore and evaluate advanced models to identify and select the most suitable model for this specific use case.

### Purpose
Detecting fraud in credit card transactions aims to minimize financial losses for a financial institution and foster trust with its customers.

### Research Inquiry
Determining whether a specific credit card transaction constitutes fraud.

### Dataset Details
The dataset, accessible at https://www.kaggle.com/datasets/kartik2112/fraud-detection, consists of simulated credit card transactions encompassing both legitimate and fraudulent activities occurring between 1st January 2019 and 31st December 2020. It includes transactions made by 1000 customers across a diverse set of 800 merchants. The data was generated using the simulation tool found at https://github.com/namebrandon/Sparkov_Data_Generation.

### Approach
The classification of transactions as fraudulent or legitimate in machine learning was conducted using the following tools and techniques.

#### Data Analysis
Exploring a dataset with over 1 million rows and more than 20 features involved detailed analysis using various visualization tools such as *matplotlib*, *seaborn*, and *plotly* to understand each feature's characteristics and their interrelationships.

Below are key findings from the analysis:

![Fraudulent Transactions by Time](images/fraud_by_time.png)

This plot indicates that fraudulent transactions are more prevalent during nighttime hours.

![Distribution of Fraud by Category](images/fraud_distri_cat.png)

The distribution plot highlights a higher incidence of fraud in internet transactions.

![Fraudulent vs Non-Fraudulent Transactions](images/fraud_vs_not.png)

This plot reveals that out of nearly 1.3 million records, only 7506 transactions are classified as fraudulent.

#### Data Preprocessing
* Removed unnecessary columns such as transaction numbers, merchant names, customer names, etc., as they do not contribute to model evaluation.

#### Feature Engineering
* Introduced additional features such as transaction hour derived from transaction datetime, and age calculated from date of birth.
* Utilized *JamesSteinEncoder* for encoding categorical data.
* Applied *StandardScaler* to standardize the dependent features.
* Split the data into training and testing sets, with a test split size of 30%.

Below are visualizations related to feature engineering:

![Correlation Heatmap](images/heatmap.png)
The heatmap displays correlations, both positive and negative, among various features. Notably, latitude and longitude with merchant coordinates show a strong positive correlation. Amount correlates positively with fraud, while transaction time and age exhibit negative correlations.

![Feature Importance](images/feature_imp.png)
This plot highlights important features crucial for predicting fraud in transactions. Key features such as amount, transaction time, and city are identified as highly influential.

#### Model Evaluation
Conducted evaluation on the following models:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Support Vector Machine (SVM)
* Gaussian Naive Bayes (GaussianNB)
* Stochastic Gradient Descent Classifier (SGDClassifier)


### Results
Following training using **GridSearchCV** with 5-fold cross-validation, the models yielded the following outcomes:

None of these models accurately predicted *true positive* classes.

Below are confusion matrix plots for the evaluated models:

![](images/log_reg_matrix.png) ![](images/sgd_matrix.png) ![](images/decision_tree_matrix.png) ![](images/guassian_matrix.png) ![](images/knn_matrix.png)

These plots indicate very low *true positive* predictions across the models.

Additionally, the following plot displays the f1_score and balanced score of these models:

![](images/models_base_perf.png)

### Next Steps
While the models demonstrate higher overall classification scores, the primary challenge remains the significantly low true positive predictions, suggesting a bias towards classifying transactions as non-fraudulent. To mitigate this issue and improve model accuracy, we can explore the following techniques and/or algorithms:

Here are some additional approaches to enhance the model's performance.

#### Addressing Class Imbalance
The target class (is_fraud) is highly imbalanced, with fraud transactions accounting for only 0.5% of the entire dataset comprising 1.2 million records.

To balance the dataset, consider employing the following techniques:
* Random Over Sampling
* BorderlineSMOTE
* SMOTE
* ADASYN

These methods can help mitigate the imbalance issue and improve the model's ability to correctly classify fraudulent transactions.

#### Utilizing Ensemble Models
Ensemble techniques and models could potentially enhance the accuracy of our predictions. Consider exploring the following ensemble methods:

* BaggingClassifier
* AdaBoostClassifier
* BalancedRandomForestClassifier
* RUSBoostingClassifier
* RandomForestClassifier

These ensemble approaches leverage multiple base models to collectively improve predictive performance, which may be beneficial in addressing the challenges observed in the current models.

#### Neural Networks
Implementing deep neural network models such as *Dense* networks in conjunction with cross-validation could further enhance model accuracy.

### Outline of project

#### Project Structure
The project is organized into several folders containing various files:

* **data**: Contains train and test datasets.
* **images**: Includes various images of data analysis and model performance plots.
* **jupyter notebook**: Contains notebooks used for data exploration, model training, and evaluation.
* **README**: Provides project information and instructions.
* **git helper files**: Includes files related to version control management.

#### Deliverables
- [Jupyter Notebook](credit_card_fraud_detection.ipynb)

#### Required Libraries
To execute this project, ensure the following libraries are installed:

* Default Anaconda environment packages
* Category Encoders

These libraries are essential for running the project smoothly.
