#! /bin/bash
  
######## Part 1 #########
# Script parameters     #
#########################
 
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
 
# Specify the QOS, mandatory option
#SBATCH --qos=normal
 
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=junogpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=anti-n0-mc
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=2

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=job_output
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=32768
  
# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1
    
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
  
# list the allocated hosts
srun -l hostname
  
# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
  
sleep 1

cat /usr/local/cuda/version.txt

. ~yuansc/deploy_conda.sh hpcfs
source ~yuansc/.bashrc
setup_conda

PROJECT_DIR=/hpcfs/juno/junogpu/yuansc/dl_project_templete
python $PROJECT_DIR/scripts/run.py \