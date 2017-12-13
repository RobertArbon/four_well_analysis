#!/bin/bash


for f in 000.5pc  001.9pc  007.1pc  026.6pc  100.0pc 
do
   cd $f
   name=config
   sbatch --export=config=$name,num=100 --job-name=$f submit.sh
   cd ../
done
