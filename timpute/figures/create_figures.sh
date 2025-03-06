#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o timpute/figures/revision_cache/joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=2:00:00,h_data=4G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
#$ -M $ethanhung11@ucla.edu
# Notify when
#$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
# Edit the line below as needed:
module load mamba
module load gcc/7.5.0
module load python/3.11.5
export PATH=$PATH:~/.local/bin
echo "path exported"

# substitute the command to run your code in the two lines below:
poetry run python -m timpute.figures.figure2
poetry run python -m timpute.figures.figure3
poetry run python -m timpute.figures.figure4
poetry run python -m timpute.figures.figure5
poetry run python -m timpute.figures.supplements
poetry run python -m timpute.figures.figure6
# submit job from ./tensor-impute using: qsub timpute/figures/generate_data.sh
# check status using: myjobs

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####