# Time Series: Set of Observations Taken Sequentially over Time

## Types of time series

   - Regular time series: observations coming in at regular intervals of time
   - Irregular time series: do not have observations at a regular interval of time
    
## Main Areas of Application

   - Time series forecasting: predicting the future values of a time series, when past values are given
   - Time series classification: predict an action based on past values
   - Interpretation and causality: understand the interrelationships among several related time series
    
## Data-Generating Process (DGP)

   - Generating synthetic time series: generate time series using a set of fundamental building blocks
        - White noise: an extreme case of a stochastic process, a sequence of random numbers with zero mean and constant standard deviation
        - Red noise: a sequence of random numbers with zero mean and constant variance but is serially correlated in time
        - Cyclical or seasonal signals: most common signals
        - Autoregressive signals: popular signal in the real world, outlined as follows;
          - number of previous timesteps the signal is dependent on
          - coefficients to combine the previous timesteps
        - Mix and match: using different components to make DGP to create time series
   - Stationary time series: probability distribution remains the same at every point in time
   - Non-stationary time series: most real world data, when stationary assumption broken, have two ways to verify this;
        - Change in mean over time: mean across two windows of time would not be the same
        - Change in variance over time: variance keeps getting bigger and bigger with time, means Heteroscedasticity

## Predictability: three main factors to create a predictive model

   - Understanding the DGP: better understanding of  the DGP, higher the predictability
   - Amount of data: more data, better predictability
   - Adequately repeating pattern: more repeatable the pattern, better predictability
        
## Forecasting Terminology

   - Forecasting: prediction of future values of a time series using the known past values of the time series
   - Multivariate forecasting: multivariate time series is not only dependent on its past values but also has some dependency on the other variables. Multivariate forecasting is a model that captures the interrelationship between the different variables along with its relationship with its past and forecast all the time series together in the future
   - Explanatory forecasting: uses information other than its own history
   - Backtesting: using the history to evaluate a trained model
   - In-sample and out-sample: in-sample referring to metrics calculated on training data, and out-sample referring to metrics calculated on testing data
   - Exogenous variables: not affected by other variables, help to create the model for the target outcome
   - Endogenous variables: target variable, entirely dependent on other variables
   - Forecast combination: combine multiple forecasts
