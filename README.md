# Call_Prediction


Call Prediction is a machine learning project to predict customers that will call about
Tech Issues. 



##### The pipeline contains three main components:
1. __dataOPs__: A wrapper to preform spark operations and load data into dataframes before Preprocessor
2. __PreProcessor__: Feature engineering preprocessing from raw tables: <br />
                    - App_dcc <br />
                    - HEM <br />
                    - IPTV <br />
                    - ICM <br />
                    - IUM <br />
                    - Verint <br />
                    - Boldchat <br />

3. __Pipeline__: Including Pipeline Configurations
                - Call Prediction Pipeline (The principle pipeline configuration)
                - Training Pipeline
                - Batch Inferencing Pipeline
                - Auto Training pipeline 



### The Call Prediction Model
The latest Call Prediction Model is a RandomForest Model. It is trained on
modem data from the HEM tables. The features are aggregations over a past window period. Different statistics
are computed to give a representation of the past data. This past data is then used to predict which customers
will call for Tech Issues.


## Setting Up The Development Environment

### 1. Cloning the repository
You would first need to clone this repository to the host you want to set up your development environment.
You can use your own local environment (Rogers laptop), or an Azure ML compute instance as your
development environment host.
* __You can connect your local host to Rogers GitHub via ssh ONLY__, please follow the steps specified
[here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) to do so, prior to cloning the repository. 

### 2. Importing the project to IDE
You can import your project to an IDE of your choice, such as PyCharm or VSCode. We recommend using PyCharm
if your setting up the environment on your own Rogers laptop, and VSCode if you are setting up
your development environment on an AML compute instance. Download and install [PyCharm community edition](https://www.jetbrains.com/pycharm/download/#section=mac)
if you don't have it on your local host.

* The project has to be connected to a virtual python environment, containing CD4MT dependencies only.
We recommend using ANACONDA. Install Anaconda from [here](https://www.anaconda.com/products/individual) if you
  need to.
  
__Please make sure to use <ins> python 3.7 </ins>__





### 3. Installing Dependencies
All dependencies are included in the requirements.txt file in the staging branch. 

```shell
git checkout staging
conda create --name cp_env python=3.7.10
conda activate cp_env
pip install -r requirements.txt
```
 
* Execute the setup.py file to install all required dependencies and package your project. You 
must be at the project root directory while executing the setup.py:
```shell
pip install -e .
```

### 4. Azure Config file
For each environment you need to download the appropriate azure config file from
Azure ML and place under the conf folder. Communicating with AML requires a conf file.

### 5. Execute test scripts 
Done! Test your project by running the scripts provided at main_scripts/test_scripts
```shell
 python main_scripts/test_scripts/automl_connect_test.py 
```
If this test script runs correctly, your environment is all setup.



## The Batch Inference Pipeline Workflow 

The Batch Inference Pipeline has 2 steps:

1. ETL Step: This is a DataBricks Step where we preform the etl to create the features before sending it to the inferencing.
1. Inferencing and Save Step: This is a Python Script Step where we load the model from AML model registery 
and apply inferencing and save in some data storage.
   
The configuration yaml file can be found in conf/pipeline/inferencing

