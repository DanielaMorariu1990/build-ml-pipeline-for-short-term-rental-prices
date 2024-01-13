# Build an ML Pipeline for Short-Term Rental Prices in NYC
This project is part of the Udacity course 'MLOps engineer". 
In this project I explore the end to end development and deployment of a model pipeline using mlflow, hydra and weights&biases. 
The project development can be seen in weights&biases: https://wandb.ai/daniela-morariu1990/nyc_airbnb?workspace=user-daniela-morariu1990

The project contains the following steps:
- *eda*: I do some basic EDA, where I explore the data, using ydata pandas profiling function.
- *basic cleaning*: clean data (exclude outliers, drop rows which are not in the correct geospacial dimension etc)
- *data check*: I run some statistical test and some simple assertions of the input data to determine if it has deviated too much from the refrence data set, we have originally trained on
- *train random forest*: I train the random forest model on the data set. Using hydra to find the best hyper parameters, I release the best performing model to production, bu tagging it in wandb with the tag "prod"
- finally I *release* this pipeline on GitHub 

You can use the final version of the pipleine, by executing the command below. 
```bash
> mlflow run https://github.com/DanielaMorariu1990/build-ml-pipeline-for-short-term-rental-prices.git \
             -v 1.0.2 \
             -P hydra_options="etl.sample='sample2.csv'"
```

## License

[License](LICENSE.txt)
