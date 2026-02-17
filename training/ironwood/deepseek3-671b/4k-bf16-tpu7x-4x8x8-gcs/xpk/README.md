# Instructions for training DeepSeek3-671B on TPU Ironwood (tpu7x-4x8x8) with Google Cloud Storage (GCS)

## GCS Bucket setup
1. Create two buckets: one to hold the dataset and one to use for checkpoints. To create regional HNS buckets use the following commands:
```
# Set variables
export DATASET_BUCKET="dataloading-bucket-name"
export CHECKPOINT_BUCKET="checkpoint-bucket-name"
export REGION="us-central1"

# Create dataset bucket
gcloud storage buckets create gs://${DATASET_BUCKET} --location=${REGION}  --default-storage-class=Standard --enable-hierarchical-namespace --uniform-bucket-level-access

# Create checkpoint bucket  
gcloud storage buckets create gs://${CHECKPOINT_BUCKET} --location=${REGION}  --default-storage-class=Standard --enable-hierarchical-namespace --uniform-bucket-level-access
```
Replace the following values:  
- `<DATASET_BUCKET>`:the name of your Cloud Storage bucket with training dataset. Do not include the gs:// prefix  
- `<CHECKPOINT_BUCKET>`: the name of your Cloud Storage bucket where checkpoints will be written. Do not include the gs:// prefix
- `<REGION>`: the region where your GKE cluster is located ([available locations](https://cloud.google.com/storage/docs/locations#location-r))

2. Prepare your dataset in the DATASET_BUCKET. This recipe is configured to use the Grain loader with ArrayRecord files. Ensure your dataset files are accessible in this bucket.

## XPK setup
1. Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8/xpk/README.md) to create your GKE cluster with XPK.
2. GCSFuse lets you mount and access Cloud Storage buckets as local file systems, so applications can read and write objects in your bucket using standard file system semantics. You'll need to use the below commands to create [XPK storage resources](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#storage) for both the dataset and checkpoint buckets in order to mount them to the MaxText workload using GCSFuse. For the dataset bucket and checkpoint bucket use separate manifest files `checkpoint_pvc.yaml` and `dataset_pvc.yaml` from this repo.
Be sure to update `volumeHandle` in the yamls with your correct bucket names. Creating a bucket and xpk storage is a one time setup.
```
export RECIPE_REPO="path-to-this-recipe-repo" # Update

cd ~/xpk

python3 xpk.py storage attach dataset-bucket --type=gcsfuse --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mountpoint=/tmp/dataset --readonly=false --bucket=$DATASET_BUCKET --size=64 --automount=false --manifest=$RECIPE_REPO/tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8-gcs/xpk/dataset_pvc.yaml

python3 xpk.py storage attach checkpoint-bucket --type=gcsfuse --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mountpoint=/tmp/ckpt --readonly=false --bucket=$CHECKPOINT_BUCKET --size=64 --automount=false --manifest=$RECIPE_REPO/tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8-gcs/xpk/checkpoint_pvc.yaml
```


## Prep for MaxText

### Install MaxText and Build Docker Image
Please follow this [link](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8/MAXTEXT_README.md) to install maxtext and build the docker image.

In step 2, use the jax-stable-stack image containing JAX 0.5.2:
```
BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
```

## Run DeepSeek3-671B Workload on GKE

### Configuring and Starting workload

From the MaxText root directory, start your DeepSeek3-671B workload.

Edit the Recipe (run_recipe.sh) and populate the exported variables at the top of the file to match your environment.

```
# In run_recipe.sh, update these lines:
export PROJECT_ID="your-project-id"
export CLUSTER_NAME="your-cluster-name"
export ZONE="your-zone"
export BASE_OUTPUT_DIR="gs://${CHECKPOINT_BUCKET}"
export DATASET_BUCKET="${DATASET_BUCKET}" # e.g. "my-dataset-bucket"
export DATASET_BUCKET_MOUNTED_PATH="/tmp/dataset" # Ensure this matches where XPK mounts the bucket
```

Run the Script:
```
chmod +x run_recipe.sh
./run_recipe.sh
```

### Workload Details

For reference, here are the deepseek3-671b workload details configured in the recipe:

```
  MaxTextModel(
        model_name="deepseek3-671b",
        tuning_params={
            "per_device_batch_size": 8.0,
            "max_target_length": 4096,
            "dcn_pipeline_parallelism": 1,
            "dcn_data_parallelism": -1,
            "ici_pipeline_parallelism": 1,
            "ici_fsdp_transpose_parallelism": 2,
            "ici_fsdp_parallelism": -1,
            "allow_split_physical_axes": True,
            "use_iota_embed": True,
            "remat_policy": "custom",
            "decoder_layer_input": "offload",
            "opt_type": "adamw",
            "mu_dtype": "bfloat16",
            "grad_dtype": "bfloat16",
            "use_random_routing": True,
            "megablox": True,
            "sparse_matmul": True,
            "use_custom_sort_vjp": True,
            "shard_exp_on_fsdp": False,
            "use_2d_fsdp_sharding": True,
            "sa_use_fused_bwd_kernel": True,
            "sa_block_q": 2048,
            "sa_block_kv": 2048,
            "sa_block_q_dkv": 2048,
            "sa_block_kv_dkv": 2048,
            "sa_block_kv_dkv_compute": 2048,
            "sa_block_kv_dq": 2048,
            "sa_block_q_dq": 2048,
            "attention": "flash",
            "use_tokamax_splash": True,
            "use_max_logit_estimate": -1,
            "cost_estimate_flops_fwd": 5000000000000,
            "cost_estimate_flops_bwd": 5000000000000,
            "float32_weight_sum": False,
            "use_tokamax_gmm": True,
            "tokenizer_path": "deepseek-ai/DeepSeek-V3-Base",
            "tokenizer_type": "huggingface",
            "enable_checkpointing": True,
            "checkpoint_storage_concurrent_gb": 400,
            "async_checkpointing": True,
            "enableSingleReplicaCkptRestoring": True,
            "checkpointStorageTargetDataFileSizeBytes": 209715200,
            "dataset_type": "grain",
            "grain_file_type": "arrayrecord",
            "grain_train_files": "${DATASET_BUCKET_MOUNTED_PATH}",
            "grain_worker_count": 2,
            "steps": 30,
            "checkpoint_period": 25,
            "base_output_directory": "${BASE_OUTPUT_DIR}",
            "run_name": "${WORKLOAD_NAME}",
        },
        xla_flags=(
             "--xla_tpu_dvfs_p_state=3 "
             "--xla_tpu_scoped_vmem_limit_kib=65536 "
             "--xla_tpu_bf16_emission_mode=NATIVE_EMISSION "
             "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true "
             "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true "
             "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true "
             "--xla_tpu_enable_all_gather_offload_tracing=true "
             "--xla_tpu_use_tc_device_shape_on_sc=True "
             "--xla_sc_disable_megacore_partitioning=True "
             "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false "
             "--xla_enable_async_all_gather=true "
             "--xla_tpu_prefer_async_allgather_to_allreduce=true "
             "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true "
             "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true "
             "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true "
             "--xla_tpu_use_single_sparse_core_for_all_gather_offload=true "
             "--xla_tpu_enable_concurrent_sparse_core_offloading=true "
             "--xla_tpu_aggressive_opt_barrier_removal=true "
             "--xla_tpu_enable_offloading_gather_to_sparsecore=true "
             "--xla_tpu_sparse_core_all_gather_latency_multiplier=1 "
             "--xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 "
             "--xla_tpu_enable_sparse_core_collective_aggregator=true "
             "--xla_tpu_enable_latency_hiding_layer_scheduler=true "
             "--xla_tpu_scheduler_percent_shared_memory_limit=150 "
             "--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true "
             "--xla_tpu_pcie_bandwidth_multiplier=0.03 "
             "--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false "
             "--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true "
             "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true "
             "--xla_tpu_enable_3d_reduce_scatter_decomposer=false "
        ),
  )
```

## Clean-up

To delete the workload and free up resources:

```
# Delete the specific workload
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}

# Or delete the entire cluster
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```

You can run the following commands to detach the XPK storage resources (this removes the PersistentVolumes and PersistentVolumeClaims created by the `xpk storage attach` commands from your GKE cluster).
```
# Detach dataset storage
python3 xpk.py storage detach dataset-bucket \
  --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE

# Detach checkpoint storage
python3 xpk.py storage detach checkpoint-bucket \
  --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE
```