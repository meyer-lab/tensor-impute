#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l h_rt=72:00:00,h_data=3G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 100
# Email address to notify
#$ -M $ethanhung11@ucla.edu
# Notify when
#$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
# /u/home/e/ehung
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:

## substitute the command to run your code
## in the two lines below:

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####