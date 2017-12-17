#!/bin/bash


for f in `ls config-*.yaml`
do
   name=${f%.yaml}
   echo $name
   sbatch --export=config=$name,num=100 --job-name=$name submit.sh
done
