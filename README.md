# Trainer Package: Train Image Classification Model
This project shows how a *trainer package* can be developed and used for model training with HAFNIA's 
Training as a Service (Training aaS). 

This particular trainer package defines training of an image classification model that works on the following datasets: 
mnist, cifar10, cifar100, caltech-101 and caltech-256.

It demonstrates how to:
1) Locally develop a trainer package, where you will be able to write, run and debug a trainer package
on a sample dataset on your own machine.
2) Zip the *trainer* into a *trainer package*
3) Test the *trainer package* locally in a docker container
4) Launch trainer package with Training-aaS on the full dataset.

We will walk you through the steps.

## 1. Develop your Training Script Locally
## Setup your local environment
### Clone the Trainer Package Repo
The first step is to clone the repo to your local environment: 

    cd [SOME_DESIRED_PATH]
    git clone https://github.com/Data-insight-Platform/trainer-classification

### Create Virtual Environment and Install Hafnia Package
Install the new and amazing 'uv' package manager. 
Follow the official installation or use below command to install on macOS or Linux.

    curl -LsSf https://astral.sh/uv/install.sh | sh

Go to the cloned repo and install dependencies in a virtual environment using uv. 

    cd trainer-classification
    uv sync

    uv pip install -e .

The command `uv sync` installs python dependencies - including the Hafnia package
called `hafnia`, which we will use later. 

### Recommendation: Install and use VS Code as IDE
You can use any IDE, but for this example we recommend VS Code. 

[Download](https://code.visualstudio.com/download) and install VS Code, open the repo folder and 
install recommended extensions. 

VS Code should be able to automatically detect the python environment. If it fails, press Ctrl+Shift+P, 
search/select "Python: Select Interpreter" and select "./.venv/bin/python". 

Restart VS Code or open a new terminal in VS Code to activate the virtual environment.

### Get your HAFNIA API key
The first step is to get your API key. Use this [guide](https://hafnia.readme.io/docs/create-an-api-key) to get it.
Copy the API Key - you will need the API key in the next step. 

### Setup your API key
In the virtual python environment (created with 'uv sync'), you have also installed 'hafnia'. 

The package includes both a Command Line Interface (CLI) and a SDK to interact with Training-aaS.

Configure your machine to access the training service: 

    # Start configuration with
    hafnia configure

    # Or in case you are outside the virtual environment run:  
    uv run hafnia configure

    # You are then prompted the following questions. 
    Profile Name [default]:   # Press [Enter] or select an optional name
    Hafnia API Key:  # Pass your HAFNIA API key
    Hafnia Platform URL [https://api.mdi.milestonesys.com]:  # Press [Enter]

Well done! Your machine is now connected to Training-aaS.
This is important to both 1) use the sample dataset and 2) to launch a training script 
in the HAFNIA cloud on the full dataset.   

### Trainer Package: Code Structure and HafniaLogger
Before we run an actual training, we should briefly have a look at the trainer package code structure and 
the HafniaLogger.

#### Trainer Package: Code Structure
The trainer package code should follow below structure. 

```
    ├── src
    │   └── trainer_classification/
    ├── scripts/
    ├── Dockerfile
    ├── .hafniaignore
    ├── pyproject.toml
    └── uv.lock
```
- `scripts/*`: Folder with the actual training script. 
  Commonly, it will just be a single script called `train.py`, but you may introduce multiple scripts. 
- `src/trainer_classification/*` (Optional): Folder for all your python helper functions and/or other dependencies used 
  by your training script. In theory, all your code could be in `train.py`, but for most projects, the training code 
  will be arranged in multiple files. 
- `Dockerfile`: This dockerfile defines the environment where your script will be executed in the training service. 
- `.hafniaignore` (Optional): File for specifying files and folders that are not included in the `trainer.zip` file using same syntax as a `.gitignore` file.  
In the next section, we will cover how the `trainer.zip` file is created. 
  If you don't provide a `.hafniaignore` file, it will revert to sensible defaults.
  If you want to exclude more or less files, you will need to create a `.hafniaignore` file.
  
- `pyproject.toml` (Optional): File with dependencies. In this example, we are using a `pyproject.toml` file for 
  the `uv` package manager. This could also be a `requirements.txt` file for pip or a `environment.yml` file 
  for conda-based environments. A requirement file is not strictly needed for a trainer package. However, it can 
  be referenced in both the docker container and in your local virtual environment, to ensure you have 
  consistent dependencies locally and in the training service. 

#### Trainer Package: HafniaLogger
The `HafniaLogger` is used for experiment tracking and is responsible for logging configuration, training and 
evaluation metrics and model artifacts. 

Check out `train.py` to see how it is initialized and the `run_train_epoch` and `run_eval` function in 
`train_utils.py` to see how it is used during training and evaluation.  

More details can be found [here](https://github.com/milestone-hafnia/hafnia?tab=readme-ov-file#getting-started-experiment-tracking-with-hafnialogger). 


After above setup, it is seamless to switch between local development and launching trainer packages to Training-aaS.

For local development, you will be executing, creating, debugging and modifying your training script 
following a common developer workflow.

You can either do debugging in VS Code or run the script from the terminal.

### Run the script in VS Code
If you want to debug the script with VS Code, click the `Run and Debug` tab in the left panel, 
select `Model Training` launcher and press F5. 
Launch configurations are defined in `.vscode/launch.json`.

### Run the script in the terminal
If you want to run the script in the terminal, you can use the following command.

    python scripts/train.py --dataset mnist

    # Or if you are outside the virtual environment of vs-code
    PYTHONPATH=src uv run scripts/train.py --dataset mnist

### Experiment data
After running an experiment, you should now have a folder called `.data` in the workspace root containing the
following: 
- `.data/datasets`: This folder contains all downloaded and cached sample datasets
- `.data/experiments`: This folder contains experiments. For each run, an experiment folder is created for storing model, checkpoints and artifacts for a given run. 

## 2. Create a Trainer Package
With a working training script, you are now ready to create a trainer package.
The trainer package is essentially a zip file containing all the necessary files to reproduce your experiment
in the Training-as-a-Service (TaaS) platform.

We recommend using the CLI command `hafnia trainer create-zip` to ensure that files are correctly 
included - and that unnecessary files are excluded. Note also the `.hafniaignore` file, that
defines which files are excluded from the trainer package zip-file.

Run below command in the root folder of this repo.

    cd trainer-classification
    hafnia trainer create-zip .

    # Or if you are outside the virtual environment in vs-code
    uv run hafnia trainer create-zip .

This command will automatically gather files and create a trainer package zip-file called `trainer.zip`
in the root folder of the repo. 


## 3. Launch Training-aaS for Model Training
With a trainer package, you can now launch a training job in the HAFNIA cloud
using either the portal or the CLI. We will go through both options.

### 3a. Launching a Trainer Package through the web portal
To launch a trainer package, open
the [experiments dashboard](https://hafnia.milestonesys.com/training-aas/experiments) and press the
"New Experiment" button. 

This will open up a new window for configuring your experiment.
Fill in the following:
- **Select Trainer Package**: In the top left corner, press `Browse files` and select the `trainer.zip` file located in the root
folder of this repo. In subsequent runs, the drop down can be used to select previously used trainer packages.
- **Experiment name**: Provide some desired name. Anything works.
- **Command**: Add your training command. For this example it would be `python scripts/train.py` or
optionally provide script arguments e.g. `scripts/train.py --batch_size 256 --learning_rate 0.00001`.
- **Select dataset**: For this trainer package, you can select any Image Classification dataset
such as mnist, caltech-101 and caltech-256. 
- **Training Configuration**: Select your desired training configuration
    - "Free Tier": To use a "Nvidia T4" instance with 16 GB GPU Memory
    - "Professional": To use an instance with four "Nvidia V100" GPUs with a total of 64 GB GPU Memory. **Note** that GPU memory is distributed across multiple GPUs (16 GB per GPU). To utilize all memory, you will need to adapt your script to support multi-GPU training.


### 3b. Create and Launch Trainer Package with the CLI
If you are often making changes to the trainer package, it becomes annoying and error prone to zip and
upload a trainer package through the portal.

Instead you can use the CLI Command `hafnia experiment create` to automatically
zip and upload the trainer package to the Training-aaS platform.

    # See your options for creating an experiment
    hafnia experiment create --help

    # Example
    cd trainer-classification
    hafnia experiment create --dataset mnist --trainer-path . 

    # Showing default options
    hafnia experiment create --dataset mnist --trainer-path . --cmd "python scripts/train.py" --environment "Free Tier"

This command will create a trainer called `classifier` using the current working directory `.`.  
The training command is `python scripts/train.py`. The model is then trained on the `mnist` dataset 
using either a "Free Tier" or "Professional" instance.

After execution, the trainer will be available in `.data/trainers` and the
launch experiment can be followed in [experiments](https://hafnia.milestonesys.com/training-aas/experiments) 
on the platform. 

## Monitor Experiments
To follow the status of experiments go to [training experiments](https://hafnia.milestonesys.com/training-aas/experiments) 
on the platform. Here you can view status, logs and the option to download the trained model for all your experiments.


## Managing your Python Environment
This project is using `uv` to manage python dependencies in the `pyproject.toml`/`uv.lock` file to ensure that 
the dependencies are consistent for both the local and the Training-aaS environment.

To ensure that dependencies are consistent for the two environments, you should 
use `uv` based commands to manage dependencies.

    # Install a new dependency
    uv add <package_name>

    # Remove a dependency
    uv remove <package_name>

## Create, Build and Run `trainer.zip` locally
In order to test trainer package compatibility with Hafnia cloud use the following command to build and 
start the job locally

```bash
    # Create 'trainer.zip' from source folder
    hafnia trainer create-zip .
    
    # Build the docker image locally from a 'trainer.zip' file
    hafnia runc build-local trainer.zip

    # Execute the docker image locally with a desired dataset
    hafnia runc launch-local --dataset mnist  "python scripts/train.py"
```

**Next steps:**
- Run multiple trainings using different hyperparameters or other image classification datasets available in 
the [data library](https://hafnia.milestonesys.com/training-aas/datasets)
- Modify this template or build or your custom training script from scratch to run it with Training-aaS.

