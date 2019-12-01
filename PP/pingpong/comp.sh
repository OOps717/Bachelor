#!/bin/bash

mpicc -Wall -O2 ping.c -o ping

for i in {1..10001..50}
    do  
        mpirun -np 2 ./ping 10001 $i >> local.csv
    done