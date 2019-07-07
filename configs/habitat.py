# Habitat configs
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"
#   For descriptions of all fields, see configs/core.py
import numpy as np


@ex.named_config
def cfg_habitat():
    uuid = 'habitat_core'
    cfg = {}
    cfg['learner'] = {
        'algo': 'ppo',               # Learning algorithm for RL agent. Currently only PPO
        'clip_param': 0.1,           # Clip param for trust region in PPO
        'entropy_coef': 1e-4,        # Weighting of the entropy term in PPO
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'gamma': 0.99,               # Gamma to use if env.observation_space.shape = 1
        'internal_state_size': 512,  # If using a recurrent policy, what state size to use (if no recurrent policy, make this small for memory savings)
        'lr': 1e-4,                  # Learning rate for algorithm
        'num_steps': 512,            # Length of each rollout
        'num_mini_batch': 8,         # Size of PPO minibatch
        'num_stack': 4,              # Frames that each cell (CNN) can see
        'max_grad_norm': 0.5,        # Clip grads
        'ppo_epoch': 8,              # Number of times PPO goes over the buffer
        'recurrent_policy': False,   # Use a recurrent version with the cell as the standard model
        'tau': 0.95,                 # When using GAE
        'use_gae': True,             # Whether to use GAE
        'value_loss_coef': 1e-3,     # Weighting of value_loss in PPO
        'perception_network': 'AtariNet',
        'test': False,
        'use_replay': True,
        'replay_buffer_size': 4096,
        'on_policy_epoch': 8,
        'off_policy_epoch': 8,
    }
    cfg['env'] = {
        'add_timestep': False,             # Add timestep to the observation
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'swap_building_k_episodes': 10,
            'gpu_devices': [0],
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 99999 # For PointNav, we skip episodes that are too "hard"
            },
            'map_kwargs': {
                'map_building_size': 22,   # How large to make the IMU-based map
                'map_max_pool': False,     # Use max-pooling on the IMU-based map
                'use_cuda': False,
                'history_size': None,      # How many prior steps to include on the map
            },
            'target_dim': 16,              # Taskonomy reps: 16, scratch: 9, map_only: 1
            # 'val_scenes': ['Denmark', 'Greigsville', 'Eudora', 'Pablo', 'Elmira', 'Mosquito', 'Sands', 'Swormville', 'Sisters', 'Scioto', 'Eastville', 'Edgemere', 'Cantwell', 'Ribera'],
            'val_scenes': None,
            'train_scenes': None,
#             'train_scenes': ['Beach'],
          },
        'sensors': {
            'features': None,
            'taskonomy': None,
            'rgb_filled': None,
            'map': None,
            'target': None
        },
        'transform_fn_pre_aggregation': None,  # Transformation to apply to each individual image (before batching)
        'transform_fn_post_aggregation': None, # Transformation to apply to batched images
        'num_processes': 6,
        'num_val_processes': 1,
        'additional_repeat_count': 0,
    }

    cfg['saving'] = {
        'checkpoint':None,
        'checkpoint_configs': False,  # copy the metadata of the checkpoint. YMMV.
        'log_dir': LOG_DIR,
        'log_interval': 10,
        'save_interval': 100,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'vis_interval': 200,
        'visdom_server': 'localhost',
        'visdom_port': '8097',
    }

    cfg['training'] = {
        'cuda': True,
        'seed': 42,
        'num_frames': 1e8,
        'resumable': True,
    }


@ex.named_config
def exploration():
    uuid = 'habitat_exploration'
    cfg = {}
    cfg['learner'] = {
        'lr': 1e-3,                  # Learning rate for algorithm
        'perception_network_kwargs': {
            'n_map_channels': 1,
            'use_target': False,
        }
    }
    cfg['env'] = {
        'env_name': 'Habitat_Exploration',    # Environment to use
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'taskonomy':rescale_centercrop_resize((3,256,256)),
                    'rgb_filled':rescale_centercrop_resize((3,84,84)),
                    'map':map_pool((1,84,84)),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
        # For exploration, always map_pool in the pre_aggregation, do not in post_aggregation
        # 'map':rescale_centercrop_resize((1,84,84)),
        'transform_fn_post_aggregation': None,
        "env_specific_kwargs": {
            'scenario_kwargs': {
                'max_episode_steps': 1000,
            },
            'map_kwargs': {
                'map_size': 84,
                'fov': np.pi / 2,
                'min_depth': 0,
                'max_depth': 1.5,
                'relative_range': True,  # scale the map_range so centered at initial agent position
                'map_x_range': [-11, 11],
                'map_y_range': [-11, 11],
                'fullvision': False,  # robot unlocks all cells in what it sees, as opposed to just center to ground
            },
            'reward_kwargs': {
                'slack_reward': 0,
            }
        },
    }


@ex.named_config
def exploration_blind():
    cfg = {}
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'taskonomy': blind((8,16,16)),
                    'rgb_filled':rescale_centercrop_resize((3,84,84)),
                    'map': map_pool((1, 84, 84)),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
    }

@ex.named_config
def exploration_scratch():
    cfg = {}
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'rgb_filled':rescale_centercrop_resize((3,84,84)),
                    'map': map_pool((1, 84, 84)),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
    }

@ex.named_config
def blind():
    ''' Implements a blinded agent. This has no visual input, but is still able to reason about its movement
        via path integration.
    '''
    uuid = 'blind'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'taskonomy': blind((8,16,16)),
                    'target': identity_transform(),
                    'map': map_pool((3, 84, 84)),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
    }


@ex.named_config
def max_coverage_perception():
    ''' Implements an agent with a Max-Coverage Min-Distance Featureset
        From the paper:
            Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies
            Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
            2018
    '''
    uuid = 'habitat_max_coverage_featureset'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': 4,
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                    encoder_paths=['/mnt/models/normal_encoder.dat',
                                   '/mnt/models/keypoints2d_encoder.dat',
                                   '/mnt/models/segment_unsup2d_encoder.dat',
                                   '/mnt/models/segment_semantic_encoder.dat']),
    }


@ex.named_config
def max_coverage_perception3():
    ''' Implements an agent with a Max-Coverage Min-Distance Featureset
        From the paper:
            Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies
            Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
            2018
    '''
    uuid = 'habitat_max_coverage_featureset'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': 3,
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                    encoder_paths=['/mnt/models/edge_texture_encoder.dat',
                                   '/mnt/models/curvature_encoder.dat',
                                   '/mnt/models/reshading_encoder.dat']),
    }


@ex.named_config
def max_coverage_perception2():
    ''' Implements an agent with a Max-Coverage Min-Distance Featureset
        From the paper:
            Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies
            Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
            2018
    '''
    uuid = 'habitat_max_coverage_featureset'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': 2,
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                    encoder_paths=['/mnt/models/segment_unsup2d_encoder.dat',
                                   '/mnt/models/segment_unsup25d_encoder.dat']),
    }

@ex.named_config
def taskonomy_features():
    ''' Implements an agent with some mid-level feature.
        From the paper:
            Taskonomy: Disentangling Task Transfer Learning
            Amir R. Zamir, Alexander Sax*, William B. Shen*, Leonidas Guibas, Jitendra Malik, Silvio Savarese.
            2018
        Viable feature options are:
            []
    '''
    uuid = 'habitat_taskonomy_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_features_transform('{taskonomy_encoder}'),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                taskonomy_encoder='/mnt/models/normal_encoder.dat'),
    }


@ex.named_config
def taskonomy_decoding():
    ''' Implements an agent with some mid-level decoding.
        From the paper:
            Taskonomy: Disentangling Task Transfer Learning
            Amir R. Zamir, Alexander Sax*, William B. Shen*, Leonidas Guibas, Jitendra Malik, Silvio Savarese.
            2018
        Viable feature options are:
            []
    '''
    uuid = 'habitat_taskonomy_decoding'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 9,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                    'rgb_filled': rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'rgb_filled':cross_modal_transform(TaskonomyNetwork(load_encoder_path='{encoder}', load_decoder_path='{decoder}').cuda()),
                    'taskonomy': identity_transform(),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                encoder='/mnt/models/normal_encoder.dat',
                decoder='/mnt/models/normal_decoder.dat'),
    }



@ex.named_config
def scratch():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'habitat_scratch_map'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
             'use_target': True,
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 9,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'map':map_pool((3,84,84)),
                    'rgb_filled':rescale_centercrop_resize((3,84,84)),
                    'target':identity_transform(),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': None,
    }



@ex.named_config
def alexnet():
    ''' Implements an agent with some alexnet features. '''
    uuid = 'habitat_alexnet_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AlexNetFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 1,
            'use_target': False,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 13,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':alexnet_transform((3, 224, 224)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':alexnet_features_transform('{load_path}'),
                    'map':map_pool_collated((1,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                load_path='/mnt/logdir/models/alexnet-owt-4df8aa71.pth'),
    }

@ex.named_config
def srl_features():
    ''' Implements an agent with some alexnet features. '''
    uuid = 'habitat_alexnet_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'BaseModelAutoEncoder',
        'perception_network_kwargs': {
            'n_map_channels': 1,
            'use_target': False,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 6,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy': rescale_centercrop_resize((3,224,224)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':srl_features_transform('{load_path}'),
                    'map':identity_transform(),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                load_path='/mnt/share/midlevel_control/baselines/srl_models/HabitatPlanning/forward_inverse/srl_model.pth'),
    }




@ex.named_config
def cfg_test():
    cfg = {}
    cfg['saving'] = {
        'resumable': True,
        'checkpoint_configs': True,
    }

    override = {}
    override['saving'] = {
        'visdom_server': 'localhost'
    }
    override['env'] = {
        'num_processes': 12,
        'num_val_processes': 12,
        'env_specific_kwargs': {
            'test_mode': True,
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 99999
            }
        }
    }
    override['learner'] = {
        'test_k_episodes': 994,
        'test': True,
    }


@ex.named_config
def taskonomy_features_nomap():
    ''' Implements an agent with some mid-level feature.
        From the paper:
            Taskonomy: Disentangling Task Transfer Learning
            Amir R. Zamir, Alexander Sax*, William B. Shen*, Leonidas Guibas, Jitendra Malik, Silvio Savarese.
            2018
        Viable feature options are:
            []
    '''
    uuid = 'habitat_taskonomy_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_features_transform('{taskonomy_encoder}'),
                    'target':identity_transform(),
                    'map':blind((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
                taskonomy_encoder='/mnt/models/normal_encoder.dat'),
    }


@ex.named_config
def scratch_nomap():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'habitat_scratch_map'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
             'use_target': True,
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 9,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'map':blind((3,84,84)),
                    'rgb_filled':rescale_centercrop_resize((3,84,84)),
                    'target':identity_transform(),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': None,
    }


@ex.named_config
def blind_nomap():
    ''' Implements a blinded agent. This has no visual input, but is still able to reason about its movement
        via path integration.
    '''
    uuid = 'blind'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'taskonomy': blind((8,16,16)),
                    'target': identity_transform(),
                    'map': blind((3, 84, 84)),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
    }


all_buildings = ['Rancocas', 'Cooperstown', 'Hominy', 'Placida', 'Arkansaw', 'Delton', 'Capistrano', 'Mesic', 'Roeville', 'Angiola', 'Mobridge', 'Nuevo', 'Oyens', 'Quantico', 'Colebrook', 'Sawpit', 'Hometown', 'Sasakwa', 'Stokes', 'Soldier', 'Rosser', 'Superior', 'Nemacolin', 'Pleasant', 'Eagerville', 'Sanctuary', 'Hainesburg', 'Avonia', 'Crandon', 'Spotswood', 'Roane', 'Dunmor', 'Spencerville', 'Goffs', 'Silas', 'Applewold', 'Nicut', 'Shelbiana', 'Azusa', 'Reyno', 'Dryville', 'Haxtun', 'Ballou', 'Adrian', 'Stanleyville', 'Monson', 'Stilwell', 'Seward', 'Hambleton', 'Micanopy', 'Parole', 'Nimmons', 'Pettigrew', 'Bolton', 'Sumas', 'Sodaville', 'Mosinee', 'Maryhill', 'Woonsocket', 'Springhill', 'Annawan', 'Albertville', 'Anaheim', 'Roxboro', 'Beach', 'Bowlus', 'Convoy', 'Hillsdale', 'Kerrtown', 'Mifflintown', 'Andover', 'Brevort']

@ex.named_config
def buildings1():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:1],
            }
        }
    }


@ex.named_config
def buildings2():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:2],
            }
        }
    }


@ex.named_config
def buildings4():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:4],
            }
        }
    }


@ex.named_config
def buildings8():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:8],
            }
        }
    }


@ex.named_config
def buildings16():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:16],
            }
        }
    }



@ex.named_config
def buildings32():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:32],
            }
        }
    }



@ex.named_config
def buildings72():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings,
            }
        }
    }


@ex.named_config
def prototype():
    uuid='test'
    cfg = {}
    cfg['env'] = {
        'num_processes': 2,
        'num_val_processes': 1,
        'env_specific_kwargs': {
            'train_scenes': ['Adrian'],
            'val_scenes': ['Denmark'],
        }
    }
    cfg['saving'] = {
        'log_interval': 2,
        'vis_interval': 1,
    }


@ex.named_config
def debug():
    # this does not use VectorizedEnv, so supports pdb
    uuid='test'
    cfg = {}
    cfg['env'] = {
        'num_processes': 1,
        'num_val_processes': 0,
        'env_specific_kwargs': {
            'train_scenes': ['Adrian'],
            'debug_mode': True,
        }
    }
    cfg['saving'] = {
        'log_interval': 2,
        'vis_interval': 1,
    }


@ex.named_config
def curiosity():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'habitat_curiosity'
    cfg = {}
    cfg['learner'] = {
        'curiosity_reward_coef':0.1,
        'forward_loss_coef':0.2,
        'inverse_loss_coef':0.8,
    }

@ex.named_config
def short3m():
    uuid = 'habitat_planning_3m'
    cfg = {}
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 3
            }}}

@ex.named_config
def short5m():
    uuid = 'habitat_planning_5m'
    cfg = {}
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 5
            }}}

@ex.named_config
def short7m():
    uuid = 'habitat_planning_7m'
    cfg = {}
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 7
            }}}
