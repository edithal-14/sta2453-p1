#!/usr/bin/env bash
source /u/edithal/pyenvs/pytorch/bin/activate
python /u/edithal/git_repos/sta2453-p1/riskfuel_test.py --data_frame_name dataset/testing_data.csv > /u/edithal/git_repos/sta2453-p1/slurm_output_test.txt 2>&1
