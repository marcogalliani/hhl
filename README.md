# Harrow–Hassidim–Lloyd algorithm
The goal of this repo is to implement the Harrow–Hassidim–Lloyd algorithm to solve linear systems.

## Installation
To install the dependencies create and activate a new python environment by 
```{bash}
python -m venv .venv
source .venv/bin/activate
```
and run the following
```{bash}
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To execute code on real quantum hardware you will need an IBM API key. To do that you will need to register to this [link](https://quantum.cloud.ibm.com/signin) and generate an API key. Then, you will need it to store it in a `.env` file in the root directory following the `.env.example`.

## Implementation
The code provides an implementation of the HHL algorithm for 2-by-2 linear systems. It is structured as follows:
- `hhl_2x2_sketch.py` defines a `HHLAlgorithm` python class implementing the algorithm
- `hhl_test.py` is a python script to test the algorithm on a simple example
- `hhl_example.ipynb` adds a simpler visualisation of the algorithm

## Documentation
In parallel to the implementation a slide deck explaining the HHL algorithm step-by-step, The latex code to generate the slides is available in `docs/beamer-slides`.