#!/bin/bash 

# clean
rm -r data
rm *.json
rm discretize_UCI_by_SegDP.sh
rm UCI_data_summary.csv

# exp
python generate_ten_folds.py
chmod a+x discretize_UCI_by_SegDP.sh
./discretize_UCI_by_SegDP.sh

