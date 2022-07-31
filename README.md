# PratilipiDS_Assignment

- The Data Science project which is given here is an analysis of "Pratilipi". The project goal is to predict the "top 5 pratilipi which each user is going to read later" based on the Specification like 'published_at', 'read_percent', 'reading_time' etc. The Goal and Insights of the project are as follows,

  1. A trained model which can predict the category name of the pratilipi based on factors as inputs.


- The given data of pratilipi has the (15892133, 9) in size to perform a higher-level machine learning where it is well structured. The features present in the metadata are 954501 and in user interaction data are 1000000 in total. The Shape of the metadata is (954501, 6) and user interaction is (1000000, 5). 'pratilipi_id' is a common feature b/w these two, so we combined metadata and user interaction and make a single dataset. The 9 features are classified into quantitative and qualitative where 4 features are qualitative and 5 features are quantitative. 

- The analysis of the project has gone through the stage of distribution analysis, correlation analysis, and analysis by each department to satisfy the project goal.

-  Arrange the dataset in ascending order of reading time.

-  Use the first 75% of the data for training and evaluate your model on the next 25% of the data.

- The machine learning model which is used in this project is the Decision Tree classifier which predicted the nearby higher accuracy of 30%. 

### 1. Requirement
The data was given from the pratilipi organization for this project where the collected source is Datamites. The data is based on pratilipi. The data is from the real organization. The whole project was done in Jupiter notebook with python platform and environment.

### 2. Analysis
Data were analyzed by describing the features present in the data. the features play a bigger part in the analysis. the features tell the relation between the dependent and independent variables. Pandas also help to describe the datasets by answering the following questions early in our project. The futures present in the data are divided into numerical and categorical data.

##### Categorical Features
These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based. The categorical features are as follows, 

- category_name  
- updated_at_meta   
- published_at   
- updated_at_user   

##### Numerical Features
These values change from sample to sample. Within numerical features are the values discrete, continuous, or time-series based. The Numerical Features are as follows,
- pratilipi_id   
- reading_time
- user_id
- read_percent

##### Alphanumeric Features
Numerical, alphanumeric data within the same feature. These are candidates for correcting goals. No alphanumeric feature is present in the dataset.

##### Data Clean Check
The Data cleaning and wrangling is the part of the Data science project where the workflow of the project goes through this stage. because the damaged and missing data will lead to a disaster in the accuracy and quality of the model. If the data is already structured and cleaned, there is no need for data cleaning. In this case, the given data have some outliers, we detected and treated outliers by replacing them with mean values of respective features and making data cleaned and there are no missing data present in this data.

##### Analysis by Visualization
we can able to perform the analysis by the visualization of the data in two forms here in this project. One is by distributing the data and visualizing using density plotting. The other one is nothing but the correlation method which will visualize the correlation heat map and we can able to achieve the correlation values between the numerical features.
1. Distribution Plot
   - In general, one of the first few steps in exploring the data would be to have a rough idea of how the features are distributed with one another. To do so, we shall invoke the familiar function from the Seaborn plotting library. The distribution has been done by both numerical and categorical features. it will show the overall idea about the density and majority of data present at a different level.

2. Correlation Plot
   - The next tool in a data explorer's arsenal is that of a correlation matrix. By plotting a correlation matrix, we have a very nice overview of how the features are related to one another. For a Pandas data frame, we can conveniently use the call .corr which by default provides the Pearson Correlation values of the columns pairwise in that data frame. The correlation works best for numerical data where we are going to use all the numerical features present in the data.

##### Machine Learning Model
The machine learning model used in this project is a "decision tree"
The train and test data are divided and fitted into the model and passed through machine learning.
The predicted data and test data achieved an accuracy rate using DT is 31% 
Using a DT classifier for fitting the model and then evaluating the model.
In the model Evaluation part, we calculate,
1. accuracy score
2. confusion in matric
3. MSE and RMSE values
4. Precision
5. Recall
6. F1 score
7. Classification Report
8. Predicted Model

##### Time series forecasting
- It gives detail description about, How many user is going to read pratilipis later


### 3. Summary
The machine learning model has been fitted and predicted with the accuracy score. The goal of this project is nothing but the results from the analysis and machine learning model.

##### Goal: A trained model which can predict pratilipis (at least 5), each user is going to read based on factors as inputs.
The trained model is created using the DT classifier algorithm as follows, 
1. accuracy score is 31%
2. confusion matric, crosstab, and counter
                  
   Counter({35: 1226403, 29: 715249, 40: 480558, 20: 367851, 9: 201975, 39: 196662, 17: 119652, 44: 96808, 23: 74145, 41: 70368, 21: 49395, 26: 46291, 38: 44082, 
            25: 43109, 43: 38112, 16: 33547, 11: 30387, 1: 22225, 3: 15345, 4: 12479, 28: 12256, 12: 11076, 42: 10221, 7: 9152, 13: 9047, 34: 7444, 15: 5476, 19: 4252,             27: 3743, 33: 2724, 8: 2534, 18: 2191, 2: 2134, 36: 1785, 0: 1367, 30: 1262, 10: 676, 37: 383, 24: 191, 32: 170, 5: 120, 22: 95, 31: 67, 14: 24, 6: 1})             
   crosstab:         
             col_0	  35
    category_name	
              0	    1367
              1	    22225
              2	    2134
              3	    15345
              4	    12479
              5	    120
              6	    1
              7	    9152
              8	    2534
              9	    201975
              10	  676
              11	  30387
              12	  11076
              13	  9047
              14	  24
              15	  5476
              16	  33547
              17	  119652
              18	  2191
              19	  4252
              20	  367851
              21	  49395
              22	  95
              23	  74145
              24	  191
              25	  43109
              26	  46291
              27	  3743
              28	  12256
              29	  715249
              30	  1262
              31	  67
              32	  170
              33	  2724
              34	  7444
              35	  1226403
              36	  1785
              37	  383
              38	  44082
              39	  196662
              40	  480558
              41	  70368
              42	  10221
              43	  38112
              44	  96808
      
    Confusion Matrix:  
      array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)
                      
3. MSE value =  113.96016822408265
4. RMSE value = 10.675212795259991
5. Precision = 31%
6. Recall = 31%
7. F1 score = 31%
8. Classification Report
                       
                       precision  recall  f1-score   support

                   0       0.00      0.00      0.00      1367
                   1       0.00      0.00      0.00     22225
                   2       0.00      0.00      0.00      2134
                   3       0.00      0.00      0.00     15345
                   4       0.00      0.00      0.00     12479
                   5       0.00      0.00      0.00       120
                   6       0.00      0.00      0.00         1
                   7       0.00      0.00      0.00      9152
                   8       0.00      0.00      0.00      2534
                   9       0.00      0.00      0.00    201975
                  10       0.00      0.00      0.00       676
                  11       0.00      0.00      0.00     30387
                  12       0.00      0.00      0.00     11076
                  13       0.00      0.00      0.00      9047
                  14       0.00      0.00      0.00        24
                  15       0.00      0.00      0.00      5476
                  16       0.00      0.00      0.00     33547
                  17       0.00      0.00      0.00    119652
                  18       0.00      0.00      0.00      2191
                  19       0.00      0.00      0.00      4252
                  20       0.00      0.00      0.00    367851
                  21       0.00      0.00      0.00     49395
                  22       0.00      0.00      0.00        95

           ...
            accuracy                           0.31   3973034
           macro avg       0.01      0.02      0.01   3973034
        weighted avg       0.10      0.31      0.15   3973034
        
        
8. Predicted Model
    pratilipis(at least 5), which each user is going to read later are :'romance', 'novels', 'suspense', 'family', 'action-and-adventure'.
    

#### 4. Documentation explaining 

##### Why you choose this model :
- Decision Tree model run efficiently for large dataset.
- It uses the entire feature of the dataset.
- Takes less time in execution.

##### Explain alternatives :
- Random Forest is the best alternative model because it also runs efficiently for large datasets.
- Solve the overfitting problem
- Handle missing values and maintain the accuracy of a large proportion of data
- Combination of multiple DT.

##### How the chosen approach is better than the alternatives
- Since the random forest is a combination of multiple decision trees that's why random forest takes more time to execution, So we used DT here.
- DT is easy as compare to RF
- DT is fast & operate easy whereas RF is slow & long process.

##### Improvements to the model built
- By handling imbalance data
- By solving overfitting, underfitting problem
- By using alternative model
- By doing hyperparameter tuning
- By Using train_test_split() to get training and test sets. It control the size of the subsets with the parameters train_size and test_size. It determine the randomness of our splits with the random_state parameter. Obtain stratified splits with the stratify parameter.
