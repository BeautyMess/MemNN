#!/bin/bash
for x in 2 3 11 13 14 15 16 17 18; do
  echo $x
  time python main.py data/en/qa${x}_*_train.txt data/en/qa${x}_*_test.txt > results/q${x}.txt &
done
wait
