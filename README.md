# Recipe: Train Image Classification Model
This project shows how a *recipe* can be developed and used for model training with HAFNIA's Training as a Service (Training aaS). 

This particular recipe defines training of an image classification model that works on the following datasets: 
mnist, cifar10, cifar100, caltech-101 and caltech-256.

It demonstrates how to 1) write, run and debug a training script on a sample dataset on your own machine and 
2) zip the training script as a recipe and 3) launch the recipe in cloud on the full dataset.  

We will walk you through the steps.

## Clone the Recipe
The first step is to clone the repo to your local environment: 

    cd [SOME_DESIRED_PATH]
    git clone https://github.com/Data-insight-Platform/recipe-classification

## Install Package Manager and Install Virtual Environment
Install the new and amazing 'uv' package manager. 
Follow the official installation or use below command to install on macOS or Linux.

    curl -LsSf https://astral.sh/uv/install.sh | sh

Go to the cloned repo and install dependencies in a virtual environment using uv. 

    cd recipe-classification
    uv sync

The command `uv sync` installs python dependencies - including the HAFNIA package
called `mdi-python-tools`, which we will use later. 


## Recommendation: Install and use VS Code as IDE
You can use any IDE, but for this example we recommend VS Code. 

[Download](https://code.visualstudio.com/download) and install VS Code, open the repo folder and 
install recommended extensions. 

VS Code should be able to automatically detect the python environment. If it fails, press Ctrl+Shift+P, 
search/select "Python: Select Interpreter" and select "./.venv/bin/python". 

## Get your HAFNIA API key
The first step is to get your API key. Use this [guide](https://hafnia.readme.io/docs/create-an-api-key) to get it.
Copy the API Key - you will need the API key in the next step. 

## Setup your API key
In the virtual python environment (created with 'uv sync'), you have also installed our 'mdi-python-tools' package. 
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

## Recipe: Code Structure and MDILogger
Before we run an actual training, we should briefly have a look at the recipe code structure and 
the MDILogger.

### Recipe: Code Structure
This project folder contains many files, but only below files are needed in our recipe: 

- `src/scripts/*`: This folder contains the recipe entry point and the actual training script. 
  Commonly, it will just be a single script called `train.py`, but you may introduce multiple scripts.
- `src/lib/*` (Optional): This folder contains all your python helper functions and/or other dependencies used by your 
  training script. In theory, all your code could be in `train.py`, but for many project, the training code will arranged in multiple files. 
- `Dockerfile`: This dockerfile defines the environment where your script will be executed in the cloud. 
- `pyproject.toml` (Optional): This file basically contain python dependencies for the `uv` package manager 
  (This could also be a `requirements.txt` file for pip or a `environment.yml` file for conda-based environments). 
  A requirement file is not strictly needed for a recipe. However, it can be referenced in both 
  the docker container and in your local virtual environment, to ensure you have consistent dependencies
  locally and in the cloud. 

### Recipe: The training script 
Another important piece for your recipe is the training script. 
To use a training script in a recipe you will need to add a few lines

When a script is being initialized you should do the following:

    def main():
        batch_size = 128
        learning_rate = 0.001

        # Initialize the MDI logger
        logger = MDILogger()

        # Store checkpoints in this path
        ckpt_dir = logger.path_model_checkpoints()

        # Store the trained model in this path to make it available after training. 
        model_dir = logger.path_model()

        # Log experiment parameters
        logger.log_configuration({"batch_size": batch_size, "learning_rate": learning_rate})

        # Use the 'load_dataset' function to load a desired dataset- 
        # Locally it returns the sample dataset. In the cloud, it returns the full dataset
        dataset = logger.load_dataset("mnist")

During model training you typically also want to log losses and accuracy metrics. 

To do this, you simply add below lines in your training loop:

    # During an experiment you typically also track values during the experiment such as training loss or metrics.
    # This could look like this:
    
    logger.log_scalar("train/loss", value=0.1, step=100)
    logger.log_metric("train/accuracy", value=0.98, step=100)

    logger.log_scalar("validation/loss", value=0.1, step=100)
    logger.log_metric("validation/accuracy", value=0.95, step=100)


## Start Local Model Training
During development, it is important for the developer   
to easily run, create, debug and modify a training script locally. 

You can initiate (local) model training with the following command:

     python src/scripts/train.py --dataset mnist 

If you want to debug the script with VS Code, click the `Run and Debug` tab in the left panel, select `Model Training` launcher 
and press F5. 

After running an experiment, you should now have a folder called `.data` in the workspace root that contains the
following: 
- `.data/datasets`: This folder contains all downloaded and cached sample datasets
- `.data/experiments`: This folder contains experiments. For each run, an experiment folder is created for storing model, checkpoints and artifacts for a given run. 
- `.data/recipes`: This folder contains recipes. This folder will be empty for a clean project. 

# Launching a Recipe
To launch a recipe in the cloud, we first need to
create a recipe file as described in above sections or [here](https://hafnia.readme.io/docs/using-scripts#using-training-scripts).

We provide a few paths for launching a recipe in the cloud. 

## Launching a Recipe through the web portal
The easiest way to launch a recipe is to open 
the [experiments dashboard](https://hafnia.milestonesys.com/training-aas/experiments) and press the 
"New Experiment" button. 

This will open up a new window for configuring your experiment. 
Fill in the following: 
- **Select Recipe**: In the top left corner, press `Browse files` and select the `recipe.zip` file located in the root folder of this repo. In subsequent runs, the drop down can be used to select previously used recipes. 
- **Experiment name**: Provide some desired name. Anything works.
- **Command**: Add `train` to point to the training script in `src/scripts/train.py`. You may also add additional commands supported in `src/scripts/train.py` script, e.g. the following command `train --batch_size 256 --learning_rate 0.00001` 
- **Select dataset**: For this recipe, you can select any Image Classification dataset such as mnist, cifar10, cifar100, caltech-101 and caltech-256. 
- **Training Configuration**: Select either "Free Tier" or "Professional" as training instance.


## Create and Launch Recipe with the CLI 
If you are often making changes to the recipe, it becomes annoying and error prone to zip and upload a recipe through the portal. 

Instead, we can use a single CLI command to 1) zip training script, libraries and the dockerfile into a recipe file and 2) launch it on the platform. 

    # The command template: 
    mdi experiment create [OPTIONS] NAME SOURCE_DIR EXEC_CMD DATASET_NAME ENV_NAME

    # Example
    mdi experiment create classifier . train mnist "Free Tier"

## Model training
After a recipe has been started, you can go to [training experiments](https://hafnia.milestonesys.com/training-aas/experiments) on the platform. 

In this table view, you will be able to view running experiments, view metrics and download the model after training.
