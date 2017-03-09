# Intelligent Irrigation Module

## Description
It is a module to predict the amount of water used in irrigating a speific piece of farm land. It uses machine learning algorithms to predict daily water usage.
The necessary algorithms are implemented in python(Django web framework). This module acts as an RESTful endpoint for the smart farm platform. 

## Data attributes
Data attributes used for training and testing consist of weather attributes like precipitation, humidity, temperature, solar radiation etc.
A sample dataset can be found in the file *bills.csv*.

## Requirements
All the requirements are mentioned in the *requirements.txt* file.

## Deployment
The application is deployed using Green Unicorn(gunicorn) and *IIM.wsgi* as mentioned in the procfile.

