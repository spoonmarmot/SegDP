#!/bin/bash

# step 1: generate synthetic data set
python step1_create_synthetic_case.py
# step 2: discretize 
./step2_discretization.sh
# draw pictures
python step3_draw_discretization.py

