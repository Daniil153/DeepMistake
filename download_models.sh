#!/bin/bash
for arg in "$@"
do
  if [ "$arg" == first_concat ]; then
      wget https://zenodo.org/record/4981585/files/first_concat.zip
      unzip first_concat.zip
      rm first_concat.zip
  fi
  if [ "$arg" == mean_dist_l1ndotn_MSE ]; then
      wget https://zenodo.org/record/4992633/files/mean_dist_l1ndotn_MSE.zip
      unzip mean_dist_l1ndotn_MSE.zip
      rm mean_dist_l1ndotn_MSE.zip
  fi
  if [ "$arg" == mean_dist_l1ndotn_CE ]; then
      wget https://zenodo.org/record/4992613/files/mean_dist_l1ndotn_CE.zip
      unzip mean_dist_l1ndotn_CE.zip
      rm mean_dist_l1ndotn_CE.zip
  fi
done
