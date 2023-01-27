### srun command to submit job to SLURM cluster
srun --partition smallgpunodes --nodelist gpunode15 -c 4 --gres=gpu:1 --mem=4G /u/edithal/git_repos/sta2453-p1/slurm_script_test.sh
