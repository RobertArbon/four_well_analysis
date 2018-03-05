#!/bin/bash

for rundir in run1 run2 run3 run4 run5
do 
   for f in `ls config-*.yaml`
   do
      cp $f $rundir
      cp submit.sh $rundir
      cd $rundir
      name=${f%.yaml}
      jobname=${name: -7}
      sbatch --export=config=$name,num=100 --job-name=$jobname submit.sh
      cd ../
   done
done
