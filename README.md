
# Predictive Maintainence
The given data is a simulated engine degradation under different combinations of operational conditions and modes provided by NASA CMAPSS-Jet-Engine-Simulated-Data.

The objective is to predict the number of remaining operational cycles before failure -> Predictive Maintainence.

The data has 4 datasets, each with different Conditions and Fault Modes. Each dataset contains n number of engines, 3 Operation Settings and data from 26 sensors. The training datasets are recorded until fault. While the test datasets have data until some time before a fault. The aim is to predict the remaining operational cycles until failure.

The 4 datasets can be kept separate since they have different Conditions and Fault modes. In production, we can have 4 different models to predict RUL which are selected based on the conditions and fault modes. It is also possible to train one only model that generalizes to all 4 combinations of Conditions and Fault modes. 

## Some questions for production:
How would this predictive model(s) be used? 
How many cycles can be used as input to the model in production?   
Who is the end user?

## Development
First prioritize building an E2E pipeline with a suitable simple time-series forecasting model such as Vanilla LSTM or VAR using one only dataset. The data processing and feature engineering are kept to a minimum.
Since this is a multi-step multivariate timeseries problem, LSTM is a good choice for a model. 

Look for existing solutions to similar problems. Consider a pre-trained model. 

Two ways of development: Model Deployment, Code Deployment

# Development steps:

1. Create E2E pipeline: data ingestion, training and testing data processing, training, evaluation, inference(model on production)
    a. Create a common data processing pipeline for training and testing(inference)
        i. Data cleaning
        ii. Minimum feature engineering
            1. Linear RUL calculation
        iii. Measure time taken for processing, optimize if needed
    b. Train on Vanilla LSTM or VAR
    c. Test model
        i. Define evaluation metrics
    d. Serve model
        i. Frontend requirements
        ii. User authentication etc
    e. Write unit tests
    f. Modularize code
    g. Debug code
2. Model improvement
    a. Data pipeline
        i. Data versioning
        ii. Move to separate data pipeline if needed. Eg. DVC and S3
    b. Feature engineering
        i. Check for data correlation
        ii. Leverage domain knowledge
        iii. Stepwise RUL calculation
    c. Define further evaluation criteria if needed
    d. Separate Dev code and Prod code
    e. Training
        i. Experiment on model architecture
        ii. Experiment with data
        iii. Experiment with different algorithms
        iv. Hyperparameter tuning
    f. Develop baseline model
    g. Model selection
3. Deployment
    a. Choose deployment method
4. Model monitoring and analytics (Databricks, MLFlow)
    a. Feedback from users
        i. Missing features
        ii. Model validation
    b. Set up tests and alarms
    c. Retraining algorithm

# Important things to keep in mind:
- Follow SOLID principles for coding
- Data, Model, Code versioning
- Reproducilibity
- Write tests for data, code, model
- Run performance profiler

# References:
1. https://neptune.ai/blog/model-deployment-strategies
2. https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html 
3. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
4. https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
5. Remaining Useful Life Estimation Using Functional Data Analysis, Wang et al. https://arxiv.org/pdf/1904.06442.pdf
6. https://gallery.azure.ai/Collection/Predictive-Maintenance-Template-3
    i. https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
7. https://biswajitsahoo1111.github.io/rul_codes_open/
    i. https://github.com/biswajitsahoo1111/rul_codes_open/blob/master/notebooks/cmapss_notebooks/
    ii. CMAPSS_FD001_LSTM_piecewise_linear_degradation_model.ipynb
    iii. https://github.com/biswajitsahoo1111/rul_codes_open/blob/master/notebooks/cmapss_notebooks/CMAPSS_using_saved_model_deep_learning.ipynb


