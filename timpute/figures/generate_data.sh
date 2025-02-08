#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=72:00:00,h_data=2G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 4
# Email address to notify
#$ -M $ethanhung11@ucla.edu
# Notify when
#$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/home/e/ehung
# Edit the line below as needed:
module load mamba
module load gcc/7.5.0
module load python/3.11.5
export PATH=$PATH:~/.local/bin

# substitute the command to run your code in the two lines below:
poetry run python -m timpute.figures.figure_data 
# submit job from ./tensor-impute using: qsub timpute/figures/generate_data.sh
# check status using myjobs

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####