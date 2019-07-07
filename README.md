This repo is adopted from https://github.com/alexsax/midlevel-reps

# Docker 
To start docker:
```
docker run --runtime=nvidia -ti --rm \
    -v /tmp/.x11-unix:/tmp/.x11-unix \
    -v $PWD/runs/:/mnt/logdir/ \
    -v $PWD/configs/:/root/midlevel-reps/configs/ \
    -v $PWD/evkit/:/root/midlevel-reps/evkit/ \
    -v $PWD/scripts/:/root/midlevel-reps/scripts/ \
    -v $PWD/tnt/:/root/midlevel-reps/tnt/ \
    --network host --ipc=host \
    activeperception/habitat:1.0 bash
```

# Training
(Note: To replicate our training, increase `num_processes` )
## Planning in Habitat
i. Features
```
python -m scripts.train_rl /mnt/logdir/habitat_planning_curvature_new \
    run_training with uuid=corl_habitat_curvature \
    cfg_habitat taskonomy_features \
    cfg.saving.log_interval=10 \
    cfg.saving.logging_type=tensorboard \
    cfg.env.transform_fn_post_aggregation="TransformFactory.independent({\
        ---taskonomy---:taskonomy_features_transform(---/mnt/models/curvature_encoder.dat---),\
        ---target---:identity_transform(),\
        ---map---:map_pool_collated((3,84,84)),\
        ---global_pos---:identity_transform(),\
        },\
        keep_unnamed=False)"\
    cfg.env.num_processes=2 \
    cfg.env.num_val_processes=1 \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000 
```

ii. Scratch
```
python -m scripts.train_rl /mnt/logdir/habitat_planning_scratch_new \
    run_training with uuid=corl_habitat_planning_scratch \
    cfg_habitat scratch \
    cfg.saving.log_interval=10 \
    cfg.saving.save_interval=20 \
    cfg.saving.vis_interval=20 \
    cfg.env.num_processes=2 \
    cfg.env.num_val_processes=1 \
    cfg.saving.logging_type=tensorboard \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000
```

iii. Baselines
```
python -m scripts.train_curiosity_rl /mnt/logdir/habitat_planning_curiosity_new \
    run_training with uuid=corl_habitat_curiosity \
    cfg_habitat scratch curiosity \
    cfg.saving.log_interval=10 \
    cfg.env.num_processes=2 \
    cfg.env.num_val_processes=1 \
    cfg.saving.logging_types=tensorboard \
    cfg.learner.num_steps=600 \
    cfg.learner.replay_buffer_size=3000
```

iv. Blind
```
python -m scripts.train_rl /mnt/logdir/habitat_planning_blind_new \
    run_training with uuid=corl_habitat_planning_blind \
    cfg_habitat blind \
    cfg.saving.log_interval=10 \
    cfg.saving.save_interval=20 \
    cfg.saving.vis_interval=20 \
    cfg.env.num_processes=2 \
    cfg.env.num_val_processes=1 \
    cfg.saving.logging_type=tensorboard \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000
```


## Testing (with checkpoints)
```
python -m scripts.train_rl /mnt/logdir/habitat_planning_curvature_eval run_training with \
    uuid=corl_habitat_eval_curvature \
    cfg_test \
    cfg.saving.checkpoint=/mnt/logdir/habitat_planning_curvature \
    override.learner.num_steps=512 \
    override.env.num_processes=4 \
    override.env.num_val_processes=4 \
    override.saving.log_dir=/mnt/logdir/habitat_planning_curvature_eval/ \
    override.saving.results_log_file=/mnt/logdir/habitat_planning_curvature_eval/eval_log.pkl \
    override.saving.reward_log_file=/mnt/logdir/habitat_planning_curvature_eval/eval_rewards.pkl  \
    override.saving.visdom_log_file=None 



python -m scripts.train_rl /mnt/logdir/habitat_planning_scratch_eval run_training with \
    uuid=corl_habitat_eval_scratch \
    cfg_test \
    cfg.saving.checkpoint=/mnt/logdir/habitat_planning_scratch \
    override.learner.num_steps=512 \
    override.env.num_processes=4 \
    override.env.num_val_processes=4 \
    override.saving.log_dir=/mnt/logdir/habitat_planning_scratch_eval/ \
    override.saving.results_log_file=/mnt/logdir/habitat_planning_scratch_eval/eval_log.pkl \
    override.saving.reward_log_file=/mnt/logdir/habitat_planning_scratch_eval/eval_rewards.pkl  \
    override.saving.visdom_log_file=None 



python -m scripts.train_rl /mnt/logdir/habitat_planning_blind_eval run_training with \
    uuid=corl_habitat_eval_blind \
    cfg_test \
    cfg.saving.checkpoint=/mnt/logdir/habitat_planning_blind \
    override.learner.num_steps=512 \
    override.env.num_processes=4 \
    override.env.num_val_processes=4 \
    override.saving.log_dir=/mnt/logdir/blind/eval/ \
    override.saving.results_log_file=/mnt/logdir/habitat_planning_blind_eval/eval_log.pkl \
    override.saving.reward_log_file=/mnt/logdir/habitat_planning_blind_eval/eval_rewards.pkl  \
    override.saving.visdom_log_file=None 
```



## More commands
```
docker exec midlevel sh -c 'python -m scripts.train_rl /mnt/logdir2/habitat_planning_{task}{EXP_SUFFIX} \
    run_training with uuid=corl_habitat_{task}{EXP_SUFFIX} \
    cfg_habitat {task} {extra_cfgs} \
    cfg.saving.log_interval=10 \
    cfg.env.num_processes=8 \
    cfg.env.num_val_processes=1 \
    cfg.saving.logging_type=tensorboard \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000 \
    {LOAD_FROM} \
    cfg.env.env_specific_kwargs.gpu_devices=[0]' 




docker exec -d midlevel sh -c 'python -m scripts.train_rl /mnt/logdir2/habitat_planning_{task}{EXP_SUFFIX} \
    run_training with uuid=corl_habitat_{task}{EXP_SUFFIX} \
    cfg_habitat taskonomy_features {extra_cfgs} \
    cfg.saving.log_interval=10 \
    cfg.saving.logging_type=tensorboard \
    cfg.env.transform_fn_post_aggregation=\\"TransformFactory.independent({{\
                    ---taskonomy---:taskonomy_features_transform(---/mnt/models/{task}_encoder.dat---),\
                    ---target---:identity_transform(),\
                    ---map---:map_pool_collated((3,84,84)),\
                    ---global_pos---:identity_transform(),\
                }},\
                keep_unnamed=False)\\"\
    cfg.env.num_processes=8 \
    cfg.env.num_val_processes=1 \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000 \
    {LOAD_FROM} \
    cfg.env.env_specific_kwargs.gpu_devices=[0]'



docker exec -d midlevel sh -c 'python -m scripts.train_rl /mnt/logdir2/habitat_planning_{task}{EXP_SUFFIX} \
    run_training with uuid=corl_habitat_{task}{EXP_SUFFIX} \
    cfg_habitat taskonomy_features \
    cfg.saving.log_interval=10 \
    cfg.saving.logging_type=tensorboard \
    cfg.env.transform_fn_post_aggregation=\\"TransformFactory.independent({{\
                    ---taskonomy---:taskonomy_features_transform(---/mnt/models/{task_encoder}_encoder.dat---),\
                    ---target---:identity_transform(),\
                    ---map---:blind((3,84,84)),\
                    ---global_pos---:identity_transform(),\
                }},\
                keep_unnamed=False)\\"\
    cfg.env.num_processes=8 \
    cfg.env.num_val_processes=1 \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000 \
    {LOAD_FROM} \
    cfg.env.env_specific_kwargs.gpu_devices=[0] &'



docker exec midlevel sh -c 'python -m scripts.train_rl /mnt/logdir2/habitat_planning_{task}{EXP_SUFFIX} \
    run_training with uuid=corl_habitat_{task}{EXP_SUFFIX} \
    cfg_habitat {task} {extra_cfgs} \
    cfg.saving.log_interval=10 \
    cfg.env.num_processes=8 \
    cfg.env.num_val_processes=1 \
    cfg.saving.logging_type=tensorboard \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000 \
    {LOAD_FROM} \
    cfg.env.env_specific_kwargs.gpu_devices=[0]' 



docker exec -d midlevel sh -c 'python -m scripts.train_rl /mnt/logdir2/habitat_planning_{task}{EXP_SUFFIX} \
    run_training with uuid=corl_habitat_{task}{EXP_SUFFIX} \
    cfg_habitat srl_features {extra_cfgs} \
    cfg.saving.log_interval=10 \
    cfg.saving.logging_type=tensorboard \
    cfg.env.transform_fn_post_aggregation=\\"TransformFactory.independent({{\
                    ---taskonomy---:srl_features_transform(---/mnt/logdir/models/{task}/srl_model.pth---),\
                    ---target---:identity_transform(),\
                    ---map---:map_pool_collated((3,84,84)),\
                    ---global_pos---:identity_transform(),\
                }},\
                keep_unnamed=False)\\"\
    cfg.env.num_processes=8 \
    cfg.env.num_val_processes=1 \
    cfg.learner.num_steps=1000 \
    cfg.learner.replay_buffer_size=3000 \
    {LOAD_FROM} \
    cfg.env.env_specific_kwargs.gpu_devices=[0] '




```
