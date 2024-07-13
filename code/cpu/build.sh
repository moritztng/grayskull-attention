#!/bin/bash

gcc -O3 -march=native -ffast-math -finline-functions -v -o softmax softmax.c -lm
