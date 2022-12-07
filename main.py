from multiprocessing.dummy import Pool as ThreadPool
import argparse
import copy
import logging
import numpy as np
import os
import time
import wandb
import utils
import yaml
import tf_models

from envs.aie_env_wrapper import AIEEnvWrapper
from envs.bimatrix_env_wrapper import BimatrixEnvWrapper
from envs.coop_bimatrix_env_wrapper_3 import CoopBimatrixEnvWrapper3

import ray
from ray.rllib.agents.wppo import WPPOTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.wppo.wppo_tf_policy import WPPOTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.logger import pretty_print, NoopLogger

ray.init()


logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-dir", type=str,
                        help="Path to the directory for this run.")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log episode metrics to W&B.")

    args = parser.parse_args()
    run_dir = args.run_dir
    use_wandb = bool(args.use_wandb)

    config_path = os.path.join(args.run_dir, 'config.yaml')
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, 'r') as f:
        run_config = yaml.safe_load(f)

    return run_dir, run_config, use_wandb


def build_trainer(run_config):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_config.get('trainer')

    env_name = run_config.get('env')['env_name']
    if env_name == "toy":
        EnvWrapperCls = BimatrixEnvWrapper
    elif env_name == "cooptoy3":
        EnvWrapperCls = CoopBimatrixEnvWrapper3
    elif env_name == "aie":
        EnvWrapperCls = AIEEnvWrapper
    else:
        raise ValueError()

    # === Env ===
    env_config = {
        "env_config_dict": run_config.get('env'),
        "num_envs_per_worker": trainer_config.get('num_envs_per_worker'),
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_config['metadata']['launch_time'])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config['seed'])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): {}".format(final_seed))

    # === Multiagent Policies ===
    dummy_env = EnvWrapperCls(env_config={"env_config_dict": run_config.get('env')})

    # Policy tuples for agent/planner policy types
    if run_config["ermas"]["use_ermas"]:
        # If ERMAS is used, a cost sharing PPO is used.
        run_config["agent_policy"]["lambda_coeff"] = run_config["ermas"]["initial_lambda"]
        agent_policy_tuple = (
            WPPOTFPolicy,
            dummy_env.observation_space,
            dummy_env.action_space,
            run_config.get('agent_policy'),
        )
    else:
        agent_policy_tuple = (
            PPOTFPolicy,
            dummy_env.observation_space,
            dummy_env.action_space,
            run_config.get('agent_policy'),
        )
    planner_policy_tuple = (
        PPOTFPolicy,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_config.get('planner_policy'),
    )

    # We can add a non-training copy of agent policy for capping gradient batch sizes
    num_training_agents = int(run_config['general']['num_training_agents'])
    if num_training_agents == -1 or num_training_agents >= dummy_env.n_agents:
        # All the agents are training agents. Don't add a non-training policy.
        use_sample_policy = False
        policies = {
            "a": agent_policy_tuple,
            "p": planner_policy_tuple
        }
        policy_mapping_fun = lambda i: "a" if str(i).isdigit() or i == "a" else "p"
    else:
        # Only some of the agents are training agents. Add a non-training policy.
        assert num_training_agents >= 1
        use_sample_policy = True
        policies = {
            "a": agent_policy_tuple,
            "p": planner_policy_tuple,
            "a_sample": agent_policy_tuple,
        }
        indices_for_training = [
            int(x)
            for x in np.linspace(0, dummy_env.n_agents - 1, num_training_agents)
        ]
        policy_mapping_fun = (
            lambda i: ("a" if int(i) in indices_for_training else "a_sample")
            if str(i).isdigit()
            else "p"
        )

    # Which policies to train
    if run_config['general']['train_planner']:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    # === Finalize and create ===
    trainer_config.update({
        "env_config": env_config,
        "seed": final_seed,
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fun,
        },
        "metrics_smoothing_episodes": trainer_config.get('num_workers') *
                                      trainer_config.get('num_envs_per_worker')
    })

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    if run_config["ermas"]["use_ermas"]:
        trainer = WPPOTrainer(
            env=EnvWrapperCls, config=trainer_config, logger_creator=logger_creator
        )
    else:
        trainer = PPOTrainer(
            env=EnvWrapperCls, config=trainer_config, logger_creator=logger_creator
        )

    return trainer, use_sample_policy


def build_me_trainer(run_config, agent_idx):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_config.get('trainer')

    env_name = run_config.get('env')['env_name']
    if env_name == "toy":
        EnvWrapperCls = BimatrixEnvWrapper
    elif env_name == "cooptoy3":
        EnvWrapperCls = CoopBimatrixEnvWrapper3
    elif env_name == "aie":
        EnvWrapperCls = AIEEnvWrapper
    elif env_name == "drive":
        EnvWrapperCls = DrivingEnvWrapper
    else:
        raise ValueError()

    # === Env ===
    env_config = {
        "env_config_dict": run_config.get('env'),
        "num_envs_per_worker": trainer_config.get('num_envs_per_worker'),
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_config['metadata']['launch_time'])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config['seed'])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): {}".format(final_seed))

    # === Multiagent Policies ===
    dummy_env = EnvWrapperCls(env_config={"env_config_dict": run_config.get('env')})

    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        PPOTFPolicy,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_config.get('agent_policy'),
    )
    planner_policy_tuple = (
        PPOTFPolicy,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_config.get('planner_policy') or {},
    )

    # Only some of the agents are training agents. Add a non-training policy.
    use_sample_policy = True
    policies = {
        "a": agent_policy_tuple,
        "p": planner_policy_tuple,
        "a_sample": agent_policy_tuple,
    }
    indices_for_training = [
        int(agent_idx)
    ]
    policy_mapping_fun = (
        lambda i: ("a" if int(i) in indices_for_training else "a_sample")
        if str(i).isdigit()
        else "p"
    )

    # Which policies to train
    policies_to_train = ["a"]

    # === Finalize and create ===
    trainer_config.update({
        "env_config": env_config,
        "seed": final_seed,
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fun,
        },
        "metrics_smoothing_episodes": trainer_config.get('num_workers') *
                                      trainer_config.get('num_envs_per_worker')
    })

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    trainer = PPOTrainer(
        env=EnvWrapperCls, config=trainer_config, logger_creator=logger_creator
    )

    return trainer, use_sample_policy


def set_up_dirs_and_maybe_restore(run_dir, run_config, trainer):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
    ) = utils.saving.fill_out_run_dir(run_dir)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir {}".format(ckpt_dir)
        )

        at_loads_a_ok = utils.saving.load_snapshot(trainer, run_dir, load_latest=True)

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir {}, but no good ckpts "
                "found/loaded!".format(run_dir)
            )
            exit()

        # === Trainer-specific counters ===
        step_last_ckpt = (
            int(trainer._timesteps_total) if trainer._timesteps_total else 0
        )
        epis_last_ckpt = int(trainer._episodes_total) if trainer._episodes_total else 0


    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only tf checkpoint weights
        starting_weights_path_agents = run_config['general'].get(
            'restore_tf_weights_agents',
            ''
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents TF weights...")
            utils.saving.load_tf_model_weights(trainer, starting_weights_path_agents, remap=run_config["general"].get("remap", False))
        else:
            logger.info("Starting with fresh agent TF weights.")

        starting_weights_path_planner = run_config['general'].get(
            'restore_tf_weights_planner',
            ''
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner TF weights...")
            utils.saving.load_tf_model_weights(trainer, starting_weights_path_planner, remap=run_config["general"].get("remap", False))
        else:
            logger.info("Starting with fresh planner TF weights.")

    return (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        epis_last_ckpt,
    )

def wandb_init(run_dir, run_config):
    wandb_config = copy.deepcopy(run_config)
    if run_config["env"]["env_name"] == "aie":
        # First scrub the env Components to make better use of W&B
        if isinstance(wandb_config['env']['components'], (tuple, list)):
            components = {}
            for component in wandb_config['env']['components']:
                assert isinstance(component, dict)
                components.update(component)
            wandb_config['env']['components'] = components

    # Also remove some fields that produce weird effects
    del(wandb_config['trainer']['env_config'])
    del(wandb_config['trainer']['multiagent'])

    # Initialize W&B
    wandb_id = str(wandb_config['metadata']['expid'])
    wandb.init(
        project=wandb_config['metadata']['project'],
        tags=[wandb_config['metadata']['group']],
        entity="ericzhao28salesforce",
        dir=run_dir,
        id=wandb_id,
        resume=wandb_id if restore_from_crashed_run else False,
        config=wandb_config,
        allow_val_change=True
    )


def transfer_weights(from_trainer, to_trainer, from_policy, to_policy):
    weights = from_trainer.get_weights([from_policy])
    set_weight_dict = {
        to_policy: {
            "/".join([to_policy] + k.split("/")[1:]): v
            for k, v in weights[from_policy].items()
        }
    }
    to_trainer.set_weights(set_weight_dict)
    def init_worker(worker):
        worker.set_weights(set_weight_dict)
    to_trainer.workers.foreach_worker(init_worker)


def merge_weights(from_trainer, to_trainer, policy, from_alpha, to_alpha):
    from_weights = from_trainer.get_weights([policy])
    to_weights = to_trainer.get_weights([policy])
    set_weight_dict = {
        policy: {
            k: (from_alpha * v + to_alpha * to_weights[policy][k]) / (from_alpha + to_alpha)
            for (k, v) in from_weights[policy].items()
        }
    }
    to_trainer.set_weights(set_weight_dict)
    def init_worker(worker):
        worker.set_weights(set_weight_dict)
    to_trainer.workers.foreach_worker(init_worker)


def maybe_sync_saez_buffer(trainer, result, run_config):
    if result["episodes_this_iter"] == 0:
        return

    # This logic just detects if we're using the Saez formula
    sync_saez = False
    for component in run_config['env']['components']:
        assert isinstance(component, dict)
        c_name = list(component.keys())[0]
        c_kwargs = list(component.values())[0]
        if c_name in ["PeriodicBracketTax"]:
            tax_model = c_kwargs.get("tax_model", "")
            if tax_model == "saez":
                sync_saez = True
                break

    # Do the actual syncing
    if sync_saez:
        utils.remote.accumulate_and_broadcast_saez_buffers(trainer)


def maybe_store_dense_log(env_name, trainer, result, dense_log_frequency, dense_log_dir):
    if env_name == "toy":
        return
    if result["episodes_this_iter"] > 0 and dense_log_frequency > 0:
        episodes_per_replica = result["episodes_total"] // result["episodes_this_iter"]
        if (
                episodes_per_replica == 1
                or (episodes_per_replica % dense_log_frequency) == 0
        ):
            log_dir = os.path.join(
                dense_log_dir, "logs_{:016d}".format(result["timesteps_total"])
            )
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            if env_name == "aie":
                utils.saving.write_dense_logs(trainer, log_dir)
            if env_name == "drive":
                utils.saving.write_lite_logs(trainer, log_dir)
            logger.info(">> Wrote dense logs to: {}".format(log_dir))


def maybe_save(trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt):
    global_step = result['timesteps_total']

    # Check if saving this iteration
    if result["episodes_this_iter"] > 0:  # Don't save if midway through an episode.

        if ckpt_frequency > 0:
            if global_step - step_last_ckpt >= ckpt_frequency:
                utils.saving.save_snapshot(trainer, ckpt_dir, suffix="")
                utils.saving.save_tf_model_weights(
                    trainer, ckpt_dir, global_step, suffix="agent"
                )
                utils.saving.save_tf_model_weights(
                    trainer, ckpt_dir, global_step, suffix="planner"
                )

                step_last_ckpt = int(global_step)

                logger.info("Checkpoint saved @ step {}".format(global_step))

    return step_last_ckpt


def maybe_log_to_wandb(trainer, result, use_wandb):
    if use_wandb and result["episodes_this_iter"] > 0:
        # Always report at the end of an episode
        log_dict = utils.wandb.make_wandb_log(trainer, result, print_all=True)
        log_dict["timesteps_total"] = int(result['timesteps_total'])

        wandb.log(log_dict, step=log_dict["timesteps_total"])

        n_iters = result["training_iteration"]
        logger.info(
            "Main loop iter {}: logging to W&B".format(n_iters)
        )


if __name__ == "__main__":

    # ===================
    # === Start setup ===
    # ===================

    # Process the args
    run_dir, run_config, use_wandb = process_args()

    do_meta_update = False
    do_lambda_update = False
    do_me_weight_sync = False
    if run_config["ermas"]["use_ermas"] and run_config["ermas"]["use_me_trainers"]:
        run_config["trainer"]["force_evaluation"] = True
        if run_config["ermas"]["use_meta"]:
            do_meta_update = True
        if not run_config["ermas"]["freeze_alpha"]:
            do_lambda_update = True
        if run_config["ermas"]["use_me_weight_sync"]:
            do_me_weight_sync = True
    else:
        run_config["ermas"]["inner_main_iterations"] = 1
        run_config["ermas"]["inner_me_iterations"] = None

    # Create a trainer object
    trainer, use_sample_policy = build_trainer(copy.deepcopy(run_config))

    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

    # Init W&B
    if use_wandb:
        wandb_init(run_dir, run_config)

    # Grab config values
    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)

    # ======================
    # === Init Me trainers ===
    # ======================

    # Create trainers for computing unilateral deviations
    me_trainers = []
    me_use_sample_policies = []
    if run_config["ermas"]["use_ermas"] and run_config["ermas"]["use_me_trainers"]:
        for i in range(run_config["env"]["n_agents"]):
            me_trainer, me_use_sample_policy = build_me_trainer(run_config, i)
            me_trainers.append(me_trainer)
            me_use_sample_policies.append(me_use_sample_policy)

    # Initialize lambdas for ERMAS
    if run_config["ermas"]["use_ermas"]:
        lambdas = np.ones(run_config["env"]["n_agents"], dtype=np.float32) * run_config["ermas"]["initial_lambda"]


    def train_eval(x):
        trainer, iters, eval_first = x
        if eval_first:
            eval_stats = trainer._evaluate()["evaluation"]
        for _ in range(iters):
            stats = trainer.train()
        if not eval_first:
            eval_stats = trainer._evaluate()["evaluation"]
        return stats, eval_stats

    def train_only(x):
        trainer, iters, _ = x
        for _ in range(iters):
            result = trainer.train()
        return result

    # ======================
    # === Init training ===
    # ======================

    # If appropriate, synchronize the weights of the agent policies
    if use_sample_policy:
        transfer_weights(trainer, trainer, from_policy="a", to_policy="a_sample")

    for me_trainer, me_use_sample_policy in zip(me_trainers, me_use_sample_policies):
        transfer_weights(trainer, me_trainer, from_policy="a", to_policy="a")
        if me_use_sample_policy:
            transfer_weights(me_trainer, me_trainer, from_policy="a", to_policy="a_sample")

    old_result = None

    while num_parallel_episodes_done < run_config["general"]["episodes"]:

        # ===================
        # === Start train ===
        # ===================

        if me_trainers and run_config["ermas"]["parallel"]:
            logger.info("Running in parallel")
            jobs = [(t, run_config["ermas"]["inner_me_iterations"], False) for t in me_trainers]
            jobs.append((trainer, run_config["ermas"]["inner_main_iterations"], True))

            logger.info("Creating pool")
            pool = ThreadPool(len(me_trainers) + 1)
            pool_results = pool.map(train_eval, jobs)
            pool.close()
            pool.join()

            result, main_eval_stats = pool_results[-1]
            all_me_eval_stats = [r[1] for r in pool_results[0:-1]]
        else:
            if me_trainers:
                main_eval_stats = trainer._evaluate()["evaluation"]
            for _ in range(run_config["ermas"]["inner_main_iterations"]):
                result = trainer.train()
            all_me_eval_stats = []
            for me_trainer in me_trainers:
                for _ in range(run_config["ermas"]["inner_me_iterations"]):
                    me_trainer.train()
                if me_trainers:
                    all_me_eval_stats.append(me_trainer._evaluate()["evaluation"])

        # ===================
        # === Update alpha ==
        # ===================

        # Make sure we log metrics even when they do not appear recently
        if old_result:
            for i in range(run_config["env"]["n_agents"]):
                for k in ["adaptation_delta_", "old_perf_", "new_perf_", "lambda_"]:
                    result[k + str(i)] = old_result[k + str(i)]
        else:
            for i in range(run_config["env"]["n_agents"]):
                for k in ["adaptation_delta_", "old_perf_", "new_perf_", "lambda_"]:
                    result[k + str(i)] = 0

        if me_trainers:
            for i, me_eval_stats in enumerate(all_me_eval_stats):
                old_perf = main_eval_stats["agent_reward_mean"][str(i)]
                new_perf = me_eval_stats["agent_reward_mean"][str(i)]
                adapt_delta = new_perf - old_perf

                if do_lambda_update and not np.isnan(adapt_delta):
                    logger.info("Updating {} + {} * ({} - {} - {})".format(lambdas[i],
                                run_config["ermas"]["alpha_lr"], new_perf, old_perf, run_config["ermas"]["epsilon"]))
                    lambdas[i] = lambdas[i] + run_config["ermas"]["alpha_lr"] * (adapt_delta - run_config["ermas"]["epsilon"])
                    lambdas[i] = min(max(0, lambdas[i]), 100)
                result["adaptation_delta_" + str(i)] = adapt_delta
                result["old_perf_" + str(i)] = old_perf
                result["new_perf_" + str(i)] = new_perf
                result["lambda_" + str(i)] = lambdas[i]

            if do_lambda_update:
                trainer.update_lambda(lambdas)

        # ===================
        # ==== Meta step ====
        # ===================

        if do_meta_update:
            logger.info("Meta transfer")
            for i, me_trainer in enumerate(me_trainers):
                effective_lambda = min(run_config["ermas"]["metascale"] * lambdas[i], 2)
                merge_weights(me_trainer, trainer, "a", -effective_lambda, 1 + effective_lambda)

        # ===================
        # =Report statistics=
        # ===================

        num_parallel_episodes_done = result['episodes_total']
        global_step = result['timesteps_total']
        result['timesteps_this_iter'] = result['timesteps_total'] / result['training_iteration']
        curr_iter = result['training_iteration']

        logger.info(
            "Iter {}: steps this-iter {} total {} -> {}/{} episodes done".format(
                curr_iter,
                result["timesteps_this_iter"],
                global_step,
                num_parallel_episodes_done,
                run_config['general']['episodes']
            )
        )

        if curr_iter == 1 or result["episodes_this_iter"] > 0:
            logger.info(pretty_print(result))

        if run_config["env"]["env_name"] == "aie":
            maybe_sync_saez_buffer(trainer, result, run_config)
        maybe_store_dense_log(run_config["env"]["env_name"], trainer, result, dense_log_frequency, dense_log_dir)


        # === Dense logging ===

        maybe_log_to_wandb(trainer, result, use_wandb)
        # ===================
        # ==== Save here ====
        # ===================

        step_last_ckpt = maybe_save(
            trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt
        )

        # ===================
        # ==Propagate pols ==
        # ===================

        if use_sample_policy:
            transfer_weights(trainer, trainer, from_policy="a", to_policy="a_sample")

        for me_trainer, me_use_sample_policy in zip(me_trainers, me_use_sample_policies):
            if do_me_weight_sync:
                transfer_weights(trainer, me_trainer, from_policy="a", to_policy="a")
            if me_use_sample_policy:
                transfer_weights(me_trainer, me_trainer, from_policy="a", to_policy="a_sample")

        old_result = result

    # ===================
    # ==Save snapshots ==
    # ===================
    logger.info("Completing! Saving final snapshot...\n\n")
    utils.saving.save_snapshot(trainer, ckpt_dir)
    utils.saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix="agent")
    utils.saving.save_tf_model_weights(trainer, ckpt_dir, global_step, suffix="planner")
    logger.info("Final snapshot saved! All done.")
