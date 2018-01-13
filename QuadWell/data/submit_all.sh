#!/bin/bash


for f in 000.5pc  001.9pc  007.1pc  026.6pc  100.0pc 
do
   cd $f-split
   name=config
   sbatch --export=config=$name  --job-name=$f --time=24:00:00 submit.sh
   cd ../
done
