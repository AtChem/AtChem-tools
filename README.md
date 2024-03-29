# AtChem-tools
This repository brings together several tools designed to be used alongside [AtChem2](https://github.com/AtChem/AtChem2) with two main aims: 
-	To assist in the running of models through automated scripts
-	To assist in the analysis of output from AtChem2 models

Note that these tools provide greater flexibility than the tools provided as part of the AtChem2 source code, but are not designed to be used ‘out-of-the-box’. AtChemTools currently exists as a collection of Python functions that can be imported into your own custom Python scripts.

AtChemTools is intended to act as a collaborative hub for all AtChem2 users to share their tools for use alongside AtChem2. If you are an AtChem2 user, you are encouraged to **consider contributing to this repository** by suggesting additions and modifications to the existing code. If you use AtChem2 and analyse your data in a programming language other than Python, then you should still feel welcome to contribute. Get in touch and we will discuss how best to adapt the repository layout to include your contributions.

## Directory Structure

-	`AtChemTools` contains the Python functions intended to be imported into user’s own Python scripts
-	`Examples` contains a series of Python scripts intended to demonstrate common use-cases for the functions defined in the `AtChemTools` directory. Note that these scripts will not work until AtChemTools has been properly added to your Python path (as described in the Installation section).

## Installation

AtChem-tools can be installed by cloning the github repository and then adding the directory to `PYTHONPATH`. An example is given below, however you may need to alter the below commands to account for different directory structures:

```
cd ~
git clone https://github.com/AtChem/AtChem-tools/
export PYTHONPATH=$PYTHONPATH:$HOME/AtChem-tools/
```
This will allow you to import AtChem-tools in your python scripts using `import AtChemTools` or  (for example) `from AtChemTools.read_output import rate_df`.

### Package Dependencies

As well as several packages included in Python’s standard library, AtChemTools requires the import of the following packages:
-	[Pandas](https://pandas.pydata.org)
-	[Numpy](https://numpy.org)
-	[Pysolar](https://pysolar.readthedocs.io/en/latest/#)

## Building and Running Models
Automating the building and running of models can allow for the successive (or simultaneous) running of multiple models with shared behaviour. For example, you may wish to run several models with the same initial species concentrations except for initial VOC concentrations which vary between each model run. By using AtChemTools' automated model running, these models can be build and run, with the output being saved into pandas dataframes which can then be further processed, or saved to csv files.

Below is a description of a selection of functions defined within `AtChemTools/build_and_run.py` which can be imported into your python script using `from AtChemTools import build_and_run`, provided you have properly exported AtChemTools to PYTHONPATH.

### AtChemTools.build_and_run.write_config
Alters the configuration files for a given AtChem2 run directory based on the function arguments.
    `atchem2_path` : 
    !!!

### AtChemTools.build_and_run.write_model_params

### AtChemTools.build_and_run.build_model

### AtChemTools.build_and_run.run_model

### AtChemTools.build_and_run.write_build_run



## Reading Model Output
!!!
## Plotting Model Output
!!!


