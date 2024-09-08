for counter in `seq 0 1 20`
do
  sbatch extract_xvector.sh exp/resnet_ddp_adamw_batch256/ ${counter}
  sleep 120
done
