#!/bin/bash

rm -rf ./output/
mkdir output

G=100
K=10
T=3
R="standard"
A=1

echo case1
../../stratification_bydp/build/segdp -f ./data/case1.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case1_dsc.csv >> ./output/case1_dsc.log

echo case2
../../stratification_bydp/build/segdp -f ./data/case2.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case2_dsc.csv >> ./output/case2_dsc.log

echo case3
../../stratification_bydp/build/segdp -f ./data/case3.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case3_dsc.csv >> ./output/case3_dsc.log

echo case4
../../stratification_bydp/build/segdp -f ./data/case4.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case4_dsc.csv >> ./output/case4_dsc.log

echo case5
../../stratification_bydp/build/segdp -f ./data/case5.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case5_dsc.csv >> ./output/case5_dsc.log

echo case6
../../stratification_bydp/build/segdp -f ./data/case6.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case6_dsc.csv >> ./output/case6_dsc.log

echo case7
../../stratification_bydp/build/segdp -f ./data/case7.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case7_dsc.csv >> ./output/case7_dsc.log

echo case8
../../stratification_bydp/build/segdp -f ./data/case8.csv -g ${G} -k ${K} -t ${T} -m 0 -s ${R} -a ${A} -o ./output/case8_dsc.csv >> ./output/case8_dsc.log
