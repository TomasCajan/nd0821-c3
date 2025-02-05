## Codename nd0821-c3 : a Udacity capstone project in MLOps Nanodegree.

This is https://github.com/TomasCajan/nd0821-c3

Welcome to my solution of predicting the "Salary" Category variable in Census Income dataset.
This is an end-to-end CI/CD pipeline using a logistic regression classifier along with many other features.
While I spent way more time on creating it than I will ever admit and lost temper multiple times during that (Damn you s3!) it was an absolutely rewarding experience. I hardly ever learned so many different things at such a short time.
I did this capstone as the last in my Nanodegree progress and it was definitely a good choice, as it was by far the most advanced one out there.

The dataset is available here : https://archive.ics.uci.edu/dataset/20/census+income

## Features

This project contains all components of a CI/CD solution.
- Codebase - scripts for training and saving of model and associated encoders, validation of models, writing model card. 
- Inference script - prediction functionality wrapped in FastAPI, serving using the GET and POST method to greet the user or serve the predictions
- Tests - there are two test scripts - one for ML training functions and for the API, to make sure everything works as expected.
- Tracking code on GitHub
- Tracking data with DVC on AWS S3 bucket
- Automated GitHub Actions - CI of the project is made using GitHub Actions, where pytest and flake8 check the code first, only then it is allowed to the repo. Data and models are allways fetched from AWS S3.
- Automated Deployment on HEROKU - HEROKU account is linked to GitHub to wait for GitHub CI to pass, only then it deploys on the HEROKU platform.

## The actual API
The project API supports two methods:
- GET - Greets the user and gives instructions to use the /predict endpoint
- POST - Accepts JSON structure data to make a prediction.

- Availability - Application was deployed to https://tomatheroku-a89b9ce4b37c.herokuapp.com/  for the sake of evaluation, it will be there only shortly to not make much costs.

## Ownership
I have created all content in this project except for the scripts data.py and sanitycheck.py, which were created by the Udacity authors as a part of the starter code.
The original license file from Udacity starter repo is included.

## Troubleshooting
If you want to replicate this project or are in search for a solution you are stuck on in a similar project, note that there are several steps made, which you cannot see in the GitHub repo. Namely :
- You need Amazon AWS IAM user access to be set, allowing you to interact with s3 buckets.
- You need an actual AWS s3 bucket.
- You need DVC to be set up on your local CLI and to be linked with your AWS s3 bucket, specificaly by giving it your AWS keys.
- You need to understand that tracking with Git and DVC conflicts one with another, so Git first and DVC later.
- You need your AWS keys to be included as secret variables in both GitHub and Heroku, to be able to use them safely.
- In order to load data to Heroku/app from s3 you need to understand the file path structure in both s3 and Heroku. These both can be quite different from your local filestructure, so focus on it.

## Usage
Once you have set up all the accounts (GitHub including Actions, AWS IAM user, AWS s3, Heroku) and settings required, especially those mentioned in Troubleshooting above:
- use git to push the code to your GitHub repo, which should streamline it all the way to Heroku and deploy.
