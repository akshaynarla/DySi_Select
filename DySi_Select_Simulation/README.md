# DySi_Select

This repository contains the implementation of the thesis "Dynamic Situation-based selection of relevant sensor data" (MT3532). The repository is divided into individual modules. This module provides the necessary scripts for running the simulation on CARLA, as well as the script for verifying real-world portability. 

## Content

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Real-world portability](#evaluation-of-real-world-portability)
- [Running the simulation](#running-the-simulation)

## Directory Structure

```
    DySi_Select
    ├── Cam2BEV                     <- Cam2BEV module directory
    ├── SemSeg                      <- SemSeg module directory 
    ├── SIN                         <- SIN module directory
    ├── _out                        <- output frames from simulation
    │   ├── rgb                     <- output frames from RGB Camera, if used
    │   ├── sem                     <- output frames from Semantic segmentation Camera, if used
    │   └── test                    <- output frames of the entire pipeline (with situation label and relevant data)
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment
    ├── config.py                   <- Script for configuring the CARLA world (from CARLA examples)
    ├── DySi_Sel_Traffic_v01.py     <- Script for spawning traffic vehicles
    ├── DySi_Sel_utils.py           <- Script with helper functions for the CARLA clients
    ├── DySi_Sel_v01.py             <- Main script for running the simulation
    ├── DySi_Sel_Parking.py         <- Script for spawning parked vehicles in Town01,02 and 03
    ├── evaluate_real_dataset.py    <- Script for verifying real-world portability
    ├── playground_bev.py           <- Usage example of Cam2BEV interfaces
    ├── playground_sem.py           <- Usage example of SemSeg interfaces 
    ├── playground_sin.py           <- Usage example of SIN interfaces
    └── README.md                   <- Details on installation and running the simulation
```

## Installation

- Download or clone this repository from GitHub.
- It is recommended to setup the directory structure as specified above. 
- Install the necessary packages by running the requirements file given, by running the command on conda as specified on the requirements.txt file. Any other missing packages can be installed manually later.
- Once the directory is setup, you are ready to run any involved components as provided in the respective module readmes.
- The current repository is tested with CARLA 0.9.11 and 0.9.13 with both Python 3.7 and 3.8. 
    - The drawback of the current software: all components designed around the open-source Cam2BEV and uses a relatively older version of Tensorflow (tested with tensorflow-2.5.0)

It should be noted that the simulation is to be run from the correct directory, else warnings or errors are expected.

## Evaluation of Real-world portability

To verify the functionality with real-world datasets, the script `evaluate_real_dataset.py` is to be run.

- To know the parameters that can be configured in the evaluation, run `python3 evaluate_real_dataset.py --help`
- To run evaluation, just run `python3 evaluate_real_dataset.py`.
    - By default, a CityScapes dataset is configured to run. But this is possible only if the dataset is downloaded.
    - It is recommended to provide paths to image dataset folder as given in the example.
    Ex: `python3 evaluate_real_dataset.py -o <PATH TO DATASET>`
    - It is also recommended to provide the parameters weight1, backbone and weight2 from command line.
- Once the evaluation is completed, the output of the situation identification would be available in "eval_output" folder.

## Running the simulation

To run the CARLA simulation of the implemented system, an example of running the necessary steps is provided.
- It is necessary to run the Carla server before starting any simulation.
    - Run the executable file from the carla installation location by opening a terminal.
    - It is recommended to run the carla server in off-screen mode and on a low-quality setting if the GPU memory is less. Ex: `./CarlaUE4.sh -RenderOffScreen -quality-level=low`
    - This will start the server, but without any visualization yet.
- Run the config script from a new terminal to set the map to one of Town01, 02 or 03.
    - By default, carla server opens with Town10 on CARLA-0.9.13 and Town03 on CARLA-0.9.11.
    - To change the map, run `python3 config.py --map TownXX` (Replace XX with needed map number. Currently, the algorithm is tested on Town01, 02, 03 and 10)
- Open a new terminal and run the main simulation script.
    - `python3 DySi_Sel_v01.py --sync fps=20` runs the ego-vehicle at 20 FPS on the configured cameras. The sync command means that the ego-vehicle is the owner of ticks and is in control of the simulation. If not run in sync, it is expected to get undesired simulations.
- Open a new terminal and run the script to spawn traffic vehicles (only cars) randomly at various locations in the map. This script can be run in any map. 
    - `python3 DySi_Sel_Traffic_v01.py -n 75` spawns 75 cars randomly which are controlled by the Traffic Manager.
- Run the script to spawn parked vehicles in various locations of the map. This script gets executed if the map is Town01,02 or 03. Command: `python3 DySi_Sel_Parking.py`
- The output is stored on the "_out" folder.

For more details about the script, it is recommeneded to run the scripts with `--help` parameter and also read the API documentation provided (refer the pdf if the html pages do not open properly). To generate API documentation, refer the Thesis documentation.

The results and the discussion are documented in the thesis document. Please refer to the document for thesis results.


