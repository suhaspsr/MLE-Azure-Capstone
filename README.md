# Capstone Project - Azure Machine Learning Engineer

## Introduction
In this project, I train two machine learning models to perform a binary classification and compare their performance.
1. Automated ML (denoted as AutoML from now on) model: Trained using AutoML.
2. HyperDrive model: Logistic Regression with hyperparameters tuned using HyperDrive.

I demonstrate the ability to leverage an external dataset in my workspace, train the models using the 
different tools available in the AzureML framework as well as to deploy the best performing model as a web service.


## Dataset

### Overview

The Wine Quality datasets have been taken from the UCI Machine Learning Repository. The data is broken up into two 
individual datasets, one for red wines and the other for white wines. The red wine dataset contains 1599 examples while
the white wine dataset has 4898 examples. Both datasets have the same 12 attributes as follows:

Attribute information:

For more information, read [Cortez et al., 2009].

Input variables (based on physicochemical tests):
1. fixed acidity: numeric
2. volatile acidity: numeric
3. citric acid: numeric
4. residual sugar: numeric
5. chlorides: numeric
6. free sulfur dioxide: numeric
7. total sulfur dioxide: numeric
8. density: numeric
9. pH: numeric
10. sulphates: numeric
11. alcohol: numeric
12. quality: numeric

There are no missing values. I combine these datasets into one dataset and use the resulting dataset to for 
modeling in Azure. One model trained is a Logistic Regression whose hyperparameters are tuned by HyperDrive. The other 
model is trained is using AutoML. The goal for both models is to predict whether a wine is red or white. 

Response variable: (y)
13. y: (0: white, 1: red)

This dataset is public and available for research. I have included the citation below as requested:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
            [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
            [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib


### Access

The wine quality datasets live on the UCI Machine Learning Repository. I read them as a *pandas DataFrame* using the  
*read_csv* function and then store them on Azure blob storage for consumption.

*Figure 1: Wine Quality Data*
![data](images/data.png)

## Automated ML

I set the following AutoML parameters: 

* *experiment_timeout_minutes*: 30
* *enable_early_stopping*: True    
* *primary_metric*: 'accuracy'
* *n_cross_validations*:5
* *iterations*: 50
* *max_concurrent_iterations*: 4

The *experiment_timeout_minutes* (30 mins), *iterations* (50) and *enable_early_stopping* (True) are set to reduce time 
taken for model training. Enabling early stopping allows the training process to conclude if there is no considerable 
improvement to the *primary_metric* (accuracy). *n_cross_validations* is set to 5 to ensure Bias vs Variance tradeoff
and prevent overfitting.The *max_concurrent_iterations* (4) is set as we are running on compute Standard_D2_V2 which has 
4 nodes. This allows 4 jobs to be run in parallel on each node.

### Results

The best performing model from AutoML was the XGBoost Classifier with a StandardScaler. The accuracy of this model was
99.8%. This accuracy is already incredibly high however I could potentially improve it further by modifying the 
AutoML configuration settings and trying the steps below.
* Disable early stopping, increase the number of cross validations and max_iterations allowing AutoML to evaluate more 
models and avoid overfitting.
* Employ more advanced data balancing techniques to ensure both target classes have the same number of examples. In this 
case I have simply used upsampling but I could also leverage techniques such as SMOTE sampling.
* Change my target metric to Recall, Precision, F-Score,... depending on considerations of confusion matrix preferences.
* Enable featurization to generate custom features.
* Provide test_data to the AutoML configuration by enabling the parameter.
* Enable deep learning which though takes longer to compute might show improvement.


*Figure 2: AutoML Best Model*
![automl-best_model_1](images/automl-best_model_1.png)

*Figure 3: AutoML Best Model Parameters*
![automl-best_model_1](images/automl-best_model_parameters.png)

*Figure 4: AutoML Model List*
![automl-best_model_1](images/automl-best_model_2.png)

*Figure 5: AutoML Model RunDetails*
![automl-run_details](images/automl-run_details.png)

*Figure 6: AutoML Model Runs*
![automl-details_1](images/automl-details_1.png)

## Hyperparameter Tuning

I used a Logistic Regression Classifier from Sklearn. My predictive task was binary classification and this model was 
appropriate. 

I chose to use tune 3 hyperparameters:
* *C*: Inverse of regularization strength; must be a positive float. Like in support vector machines, 
smaller values specify stronger regularization.
* *max_iter*: Maximum number of iterations taken for the solvers to converge.
* *solver*: Algorithm to use in the optimization problem.

Other possible hyperparameters were solver specific and were thus omitted. 
 
I used a RandomParameterSampling instead of an exhaustive GridSearch to reduce compute resources and time. This approach 
gives close to as good hyperparameters as a GridSearch with considerably less resources and time consumed. 

I used a BanditPolicy and set the evaluation_interval to 2 and the slack_factor to 0.1. This policy evaluates the primary 
metric every 2 iteration and if the value falls outside top 10% of the primary metric then the training process will stop. 
This saves time continuing to evaluate hyperparameters that don't show promise of improving our target metric. It prevents 
experiments from running for a long time and using up resources.

I chose the primary metric as Accuracy and to maximize it as part of the Hyperdrive run. I set the max total runs as 15 
to avoid log run times as well as the *max_concurrent_runs* to 4 as I am running this experiment on Standard_D2_V2 which 
has 4 nodes. This allows 4 jobs to be run in parallel on each node.

### Results
The best performing Logistic Regression Model came in with an accuracy of 98.57% which was great however it did not 
beat the AutoML XGBoost Classifier model. 

*Figure 7: HyperDrive Model*
![hyperdrive-details_3](images/hyperdrive-details_3.png)

*Figure 8: HyperDrive Model RunDetails*
![hyperdrive-run_details](images/hyperdrive-run_details.png)

*Figure 9: HyperDrive Models Accuracy*
![hyperdrive-details_1](images/hyperdrive-details_1.png)

*Figure 10: HyperDrive Models Parameter Tuning*
![hyperdrive-details_2](images/hyperdrive-details_2.png)

## Model Deployment

As the XGBoost Classifier AutoML model was the best performing model. I proceeded to deploy this model in Azure.
I obtained the scoring script from the best run as well as used the current environment settings for the deployment. 

I used the code below to deploy the model:

*Figure 11: AutoML Model Deployment*
![automl-deployment_success](images/automl-deployment_success.png)

I tested the working endpoint via two methods:

**Python**: 
I leveraged the *requests* package to POST two JSONs to the service endpoint.

*Figure 12: AutoML Python Interaction*
![automl-deployment_test_python](images/automl-deployment_test_python.png)


**Postman**: I posted a JSON to the service endpoint using the Postman desktop application.

*Figure 13: AutoML Postman Interaction*
![automl-deployment_postman](images/automl-deployment_postman.png)

I also was able to see these interactions in the logs:

*Figure 14: AutoML Deployment Logs*
![automl-deployment_logs](images/automl-deployment_logs.png)

Once I had completed my testing I proceeded to delete the service.

*Figure 15: AutoML Deployment Deletion*
![automl-service_deletion](images/automl-service_deletion.png)

## Screen Recording

[https://youtu.be/fUItv24ryC4](https://youtu.be/fUItv24ryC4)
