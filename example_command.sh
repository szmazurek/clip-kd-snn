python scripts/train.py \
    +experiment=baseline_vit_t16 \
    dataset=cc3m_12m_wds \
    "dataset.cc3m_shard_pattern=cc3m/cc3m-train-\{0000..0575\}.tar" \
    "dataset.cc12m_shard_pattern=cc12m/cc12m-train-\{0000..1101\}.tar" \
    "hydra.run.dir=outputs/combined_${SLURM_JOB_ID}" \
    model.compile=true
