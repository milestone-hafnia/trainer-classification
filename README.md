# Recipe: Train Image Classification Model
This is an example project for creating a script2model recipe to train an image classification model with 
HAFNIA training aaS. 

It demonstrates how to 1) write, run and debug a training script on a sample dataset on your own machine and later 2)
train a model on the full dataset by dispatch the training script as a *recipe* to the cloud.  

We will walk you through the steps now.

## Clone the Recipe
The first step is to clone the example recipe (this repo) to your local environment: 

    cd [SOME_DESIRED_PATH]
    git clone https://github.com/Data-insight-Platform/recipe-classification

## Install Package Manage and Install Virtual Environment
Install the new and amazing 'uv' package manager. 
Follow the official installation or use below command to install on macOS or Linux.

    curl -LsSf https://astral.sh/uv/install.sh | sh

Go to the cloned repo and install dependencies in a virtual environment using uv. 

    cd recipe-classification
    uv sync

## Recommendation: Install and use VS-code as IDE
You can use any IDE, but for this example we recommend you to install and use vs-code.   

## Get you HAFNIA API KEY
The first step is to get your API KEY.  Use this [guide](https://hafnia.readme.io/docs/create-an-api-key) to get it.
Copy the API Key - you will need the API key in the next step. 

## Setup your API key
Alongside other dependencies, you have with 'uv sync' installed our package 'mdi-python-tools'.
The package includes both a Command Line Interface (CLI) and a SDK to interact with our platform.

Configure your machine to access our platform: 

    # Start configuration with
    mdi configure

    # Or in case you are outside the virtual environment run:  
    uv run mdi configure

    # You are then prompted the following questions. 
    Profile Name [default]:   # Press [Enter] or select an optional name
    MDI API Key:  # Pass your HAFNIA API key
    MDI Platform URL [https://api.mdi.milestonesys.com]:  # Press [Enter]

Well done! Your machine is now connected to the HAFNIA platform.
This is important to both 1) use the sample dataset and 2) to launch a training script 
in the HAFNIA cloud on the full dataset.   

## Local Model Training
During the development of a training script, it is important for the developer   
to easily run, create, debug and modify a training script locally. 

You can initiate (local) model training with the following command:

     python src/scripts/train.py --dataset mnist 

If you want to debug the script with vs-code, click the `Run and Debug` tab in the left panel, select `Model Training` launcher 
and press F5. 

After running an experiment, you should now have a folder called `.data` in the workspace root that contain the
following folders: 
- `.data/datasets`: This folder contains all downloaded and cached sample datasets
- `.data/experiments`: This folder contains experiments. For each run, an experiment folder is created for storing model, checkpoints and artifacts for a given run. 
- `.data/recipes`: This folder contain recipes. This folder will be empty for a clean project. 

# Launching a Recipe
To launch a recipe in the cloud, we first need to
create a recipe file. Details of the recipe file can be found [here](https://hafnia.readme.io/docs/using-scripts#using-training-scripts.

Essentially, it is a zip file containing libraries (`src/lib`), training scripts (`src/scripts`) and a dockerfile. 

We provide a few paths for launching a recipe in the cloud. 

## Launching a Recipe through the web portal
The easiest way to launching a recipe is to open 
the [experiments dashboard](https://hafnia.milestonesys.com/training-aas/experiments) and press the 
"New Experiment" button. 

This will open up a new window for configuring your experiment. 
Fill in the following: 
- Select Recipe: In the top left corner, press `Browse files` and select the `recipe.zip` file located in the root folder of this repo. In subsequent runs, you can simply select it from the drop down list. 
- Experiment name: Provide some desired name. Anything works.
- Command: Add `train` to point to the training script in `src/scripts/train.py`. You may also add additional commands supported in `src/scripts/train.py` script, e.g. the following command `train --batch_size 256 --learning_rate 0.00001` 
- Select dataset: For this recipe, you can select any Image Classification dataset such as mnist, cifar10, cifar100, caltech-101 and caltech-256. 

## Create and Launch Recipe with the CLI 
If you are often making changes to the recipe, it becomes annoying and error prone to zip and upload a recipe through the portal. 

Instead, we can with a single command, zip training script, libraries and the dockerfile into a recipe file and launch it on the platform. 

    # The command template: 
    mdi experiment create [OPTIONS] NAME SOURCE_DIR EXEC_CMD DATASET_NAME ENV_NAME

    # Example
    mdi experiment create classifier . train mnist "Free Tier"

Additional resources: 
- Why would I use the sample dataset and not just the whole dataset? 
Read more on sample datasets [here](https://hafnia.readme.io/docs/sample-data)
- Want to know more about script2model? Read more [here](https://hafnia.readme.io/docs/using-scripts) 