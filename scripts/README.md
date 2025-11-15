# check the status of a subset of GPUs:
```bash
condor_status  -constraint 'PartitionableSlot && Gpus > 0' -af:h Machine TotalGpus GPUs CUDADeviceName Cpus Memory/1024
```

# accept device
* NVIDIA A100-SXM4-80GB
* NVIDIA H100
* NVIDIA H100 80GB HBM3

# Envionment Setup
Refer to scripts/run.sh

# submit
```bash
condor_submit_bid 400 scripts/training.sub
```