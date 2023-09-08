# Online task-space trajectory planning using real-time estimations of robot motion capabilities 

This repo brings the comparison of the proposed Cartesian Space trajectory planning approach against [toppra](https://github.com/hungpham2511/toppra). 
The method is based on Trapezoidal Acceleration Profile (TAP) planning which is implemented using [ruckig](https://github.com/pantor/ruckig)

The method is described more in detail within the [preprint](https://inria.hal.science/hal-03791783/document).

## Installation
The code is implemented in Python and we strongly suggest to use anaconda to install the necessary libraries. 

You can install the dependencies using:
```
conda env create -f env.yaml
```

And then activate the environment 
```
conda activate planning_env
```

## Launch the code
To access the code launch the jupyter lab 
```
jupyter lab
```
and launch the notebook `comparison_with_toppra.ipynb`