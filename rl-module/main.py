'''
PyTorch Orca + MBPO + Certified Signal

One Learner: Agent, RP_model, Model (just one, TODO: multiple (ensembles of) models for different environments)
Multiple Actors:
1. collecting data from different environment instantiations to get D_env_i
2. M_i
3. \pi_i
4. D_model_i
'''

import logging
import sys
from agent_pytorch import Agent
import os
import argparse
import gym
import numpy as np
import time
import random
import datetime
import sysv_ipc
import signal
import pickle
from utils import logger, Params
from envwrapper import Env_Wrapper, TCP_Env_Wrapper, GYM_Env_Wrapper

import torch

RP_DIR = f"./rl-module/rp_dir"


def evaluate_TCP(env, agent, epoch, summary_writer, config, params, s0_rec_buffer, eval_step_counter, f_log_file):
    score_list = []

    eval_times = 1
    eval_length = params.dict['max_eps_steps']
    start_time = time.time()
    for _ in range(eval_times):
        step_counter = 0
        ep_r = 0.0

        if not params.dict['use_TCP']:
            s0 = env.reset()

        if params.dict['recurrent']:
            a = agent.get_action(s0_rec_buffer, False)
        else:
            a = agent.get_action(s0, False)
        a = a[0][0]

        env.write_action(a)

        while True:
            eval_step_counter += 1
            step_counter += 1

            s1, r, terminal, error_code = env.step(a, eval_=True)

            if error_code == True:
                s1_rec_buffer = np.concatenate((s0_rec_buffer[params.dict['state_dim']:], s1) )

                if params.dict['recurrent']:
                    a1 = agent.get_action(s1_rec_buffer, False)
                else:
                    a1 = agent.get_action(s1, False)

                a1 = a1[0][0] # why is this necessary?

                env.write_action(a1)

            else:
                print("Invalid state received...\n")
                env.write_action(a)
                continue

            ep_r = ep_r + r

            s0 = s1
            a = a1
            if params.dict['recurrent']:
                s0_rec_buffer = s1_rec_buffer

            if step_counter == eval_length or terminal:
                score_list.append(ep_r)
                break

    print(f"Eval/Return(Score) of actor {config.task}: {np.mean(score_list)}")
    f_log_file.write(f"Eval/Return(Score) of actor {config.task}: {np.mean(score_list)}\n")

    return eval_step_counter


def write_rp(f, fd_list): # actor writes one line of replay buffer to the file
    pickle.dump(fd_list, f)

def read_config():
    '''
    read the input configs for distributed training
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--eval', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--tb_interval', type=int, default=1)
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--mem_r', type=int, default = 123456)
    parser.add_argument('--mem_w', type=int, default = 12345)
    parser.add_argument('--base_path',type=str, required=True)
    parser.add_argument('--job_name', type=str, choices=['learner', 'actor'], required=True, help='Job name: either {\'learner\', actor}')
    parser.add_argument('--learner_num_steps', type=int, default = 200) 
    parser.add_argument('--task', type=int, required=True, help='Task id') # actor idx
    parser.add_argument('--training_session', type=int, default=0, help='Training session id')
    parser.add_argument('--mix_env', default=False, help="per training round, use single env or mix envs")
    parser.add_argument('--actor_max_epochs', type=int, default = 200) # per actor # epoch #TODO: should be 50k use 500 for now
    parser.add_argument('--num_ac_updates_per_step', type=int, default = 1) # per actor # step, Orca's default is 1
    parser.add_argument('--rp_dir', action='store_true', default=f"rp_dir", help='default is  %(default)s')

    ## parameters from parser
    global config
    global params

    config = parser.parse_args()
    params = Params(os.path.join(config.base_path, 'params.json'))
    return config, params


def create_files(path_list):
    for file_path in path_list:
        with open(file_path, 'w') as f:
            pass
        f.close()


def core():
    # process the parser
    config, params = read_configs()
    training_session = config.training_session
    print(f"training_session name: {training_session}")

    CKPT_DIR = f"./rl-module/training_log/safe_Orca"
    CKPT_DIR = CKPT_DIR + "_" + str(training_session)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    # config directories/parameters & initialization
    finish_file = os.path.join(CKPT_DIR, "finish.txt")
    is_learner = config.job_name == 'learner'
    def is_actor_fn(i): 
        return config.job_name == 'actor' and i == config.task
    
    # set up the TCP usage
    if params.dict['use_TCP']:
        env_str = "TCP"
        env_peek = TCP_Env_Wrapper(env_str, params, use_normalizer=params.dict['use_normalizer'])
    else:
        env_str = 'YourEnvironment'
        env_peek =  Env_Wrapper(env_str)
    
    s_dim, a_dim = env_peek.get_dims_info()
    action_scale, action_range = env_peek.get_action_info()

    if not params.dict['use_TCP']:
        params.dict['state_dim'] = s_dim
    if params.dict['recurrent']:
        s_dim = s_dim * params.dict['rec_dim']

    if params.dict['use_hard_target'] == True:
        params.dict['tau'] = 1.0
    
    # no restriction for the device
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    
    # create agent (worker_i interacts with env_i)
    # each agent comes with a \pi_i, D_env_i(replay_buffer), D_model_i(model_replay_buffer), M_i(dynamics_model)
    # each learner comes with a D_model_i, M_i, \pi_i
    # TODO: add the model replay buffer, replay buffer, and dynamics model
    agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], 
                    h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],
                    stddev=params.dict['stddev'],
                    mem_size=params.dict['memsize'],
                    gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],
                    lr_a=params.dict['lr_a'],
                    tau=params.dict['tau'],
                    PER=params.dict['PER'],
                    CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],
                    noise_type=params.dict['noise_type'],
                    noise_exp=params.dict['noise_exp'], 
                    train_dir=CKPT_DIR, 
                    num_actors=params.dict['num_actors'])

    if is_learner:
        log_file_path = f"{CKPT_DIR}/learner_log.txt"
    else:
        log_file_path = f"{CKPT_DIR}/actor{config.task}_log.txt"
    f_log_file = open(log_file_path, 'w')

    # initialize environment
    for i in range(params.dict['num_actors']):
        if is_actor_fn(i):
            if params.dict['use_TCP']:
                shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
                env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, shrmem_r=shrmem_r, shrmem_w=shrmem_w,use_normalizer=params.dict['use_normalizer'])
            else:
                env = GYM_Env_Wrapper(env_str, params)

    if is_learner:
        print("=========================Learner is up===================")
        start_time = time.time()

        signal_file_path_lists = [os.path.join(CKPT_DIR, "signal_file_" + str(i) + ".txt") for i in range(params.dict['num_actors'])]
        actor_model_rp_file_path_lists = [os.path.join(RP_DIR, "actor_model_rp_file_" + str(i) + "_" + str(training_session) + ".txt") for i in range(params.dict['num_actors'])]
        learner_id_path = os.path.join(CKPT_DIR, "learner_id.txt") # to record the learner id for the current model

        if config.load is False:
            agent.init_target()
        agent.save_model()

        create_files(signal_file_path_lists)
        
        epochs = 0
        updates_made = 0

        rollout_length = 20
        effective_model_rollouts_per_steps = 400
        freq_train_model = 64
        epoch_length = 64
        sac_batch_size = 256
        num_sac_updates_per_step = 1 # could be tuned later

        # per epoch, collect data from all the actors once
        while epochs < config.learner_epochs:
            print(f"==== Learner in Epoch {epochs}; Step {env_steps} ====")
            epoch_start_time = time.time()
            finished_actor_id_list = []
        
            # listen to the actors
            # if there is one actor finishes, collect the finished env id and randomly select one env_i to collect
            # the M_i and D_m_i (model_replay_buffer).
            # if collected, mark as collected and re-fresh the model_replay_buffer
            #TODO. or, wait for all the actors to finish
            while True:
                actor_i = 0
                for signal_file_path in signal_file_path_lists:
                    if os.path.exists(signal_file_path) is False:
                        break
                    f_actor_signal = open(signal_file_path, 'r')
                    tmp_signal = f_actor_signal.read()
                    if tmp_signal == '1':
                        finished_actor_id_list.append(actor_i)
                    else:
                        break
                    f_actor_signal.close()
                    actor_i += 1
                
                if len(finished_actor_id_list) > 0:
                    break
            
            # there is at least one actor finished,randomly select the env idx
            selected_id = random.choice(finished_actor_id_list)
            # load selected dynamics_model, model_replay_buffer
            selected_actor_model_rp_file_path = actor_model_rp_file_path_lists[selected_id]
            f_selected_model_rp = open(selected_actor_model_rp_file_path, 'rb')
            buffer_data = pickle.load(f_selected_model_rp)
            for data in buffer_data:
                agent.model_store_experience(data[0], data[1], data[2], data[3], data[4])
                # TODO: add the store_model_experience to the agent
            # TODO: load the dynamics_model with the selected one
            print(f"agent's model rp length: {agent.model_rp_buffer.length_buf}")

            # clear the selected actor's model_rp
            open(selected_actor_model_rp_file_path, 'w').close()

            ac_buffer_capacity = rollout_length * effective_model_rollouts_per_step * freq_train_model

            # agent.model_rp_buffer, agent.dynamics_model
            for _ in range(num_sac_updates_per_step):
                # always using the model data
                # TODO: agent.train()
                """ # SAC
                policy_loss = agent.sac_agent.update_parameters(
                    # agent.model_rp_buffer
                    sac_batch_size,
                    # agent.dynamics_model
                    # agent.lambda,
                    reverse_mask=True
                )
                agent.sac_agent.policy_optim.zero_grad()
                policy_loss.backward()
                agent.sac_agent.policy_optim.step()
                agent.sac_agent.update_parameters_with_alpha_loss()
                agent.sac_agent.agent_soft_update(updates=updates_made)
                updates_made += 1
                """
                agent.train_step()
                if params.dict['use_hard_target'] == False:
                        agent.target_update()
                    else:
                        if counter % params.dict['hard_target'] == 0 :
                            agent.target_update()
                

            # finish the training
            agent.save_model()

            # write learner id
            learner_id_f = open(learner_id_path, 'w')
            learner_id_f.write(str(epochs))
            learner_id_f.close()

            # clean up the finished signal files
            for signal_file_idx, signal_file_path in enumerate(signal_file_path_lists):
                if signal_file_idx in finished_actor_id_list:
                    open(signal_file_path, 'w').close()
            
            epochs += 1
            print(f"Epoch {epochs} finished. Time used: {time.time() - epoch_start_time}")
        
        # finish the training
        f_finish = open(finish_file, 'r')
        f_finish.write('1')
        f_finish.close()

    else: # if actor
        print(f"=========================Actor {config.task} is up===================")

        # append to the model_rp_buffer
        # create own env_buffer
        # read learner_id
        # read/write to the signal file
        signal_file_path = os.path.join(CKPT_DIR, "signal_file_" + str(config.task) + ".txt")
        actor_rp_file_path = os.path.join(RP_DIR, "actor_rp_file_" + str(config.task) + "_" + str(training_session) + ".txt")
        actor_model_rp_file_path = os.path.join(RP_DIR, "actor_model_rp_file_" + str(config.task) + "_" + str(training_session) + ".txt")
        actor_model_path = os.path.join(CKPT_DIR, "trained_model/dynamics_model.pth")

        ckpt_path = f"{CKPT_DIR}/trained_model/model.pth" # the policy

        print(f"ckpt exists: {os.path.exists(ckpt_path)}")
        print(f"dynamics model ckpt exists: {os.path.exists(actor_model_path)}")
        print(f"rp exists: {os.path.exists(actor_rp_file_path)}")
        print(f"model rp exists: {os.path.exists(actor_model_rp_file_path)}")

        #TODO.  NOW I try to collect all the data from the all actors, when all finishes, then training
        # listen to the status from the learner side
        actor_epoch = 0
        while True:
            print(f"Wait for the learner to finish the training...")
            while True:
                if os.path.exists(finish_file):
                    print(f"finish file exists")
                    finish_flag = True
                    break
                if os.path.exists(ckpt_path):
                    if not os.path.exists(actor_rp_file_path):
                        break
                    f_signal_f = open(signal_file_path, 'r')
                    if f_signal_f.read() == '':
                        f_signal_f.close()
                        break
                # read learner epoch
                if os.path.exists(learner_id_path):
                    f_learner_id = open(learner_id_path, 'r')
                    tmp_learner_id = f_learner_id.read()
                    if tmp_learner_id != '':
                        current_learner_epoch = int(tmp_learner_id)
                        f_learner_id.close()
                        if actor_epoch == current_learner_epoch: # consistency
                            break
                        break
            
            print(f"learner finish: {finish_flag}")
            if finish_flag:
                break

            print(f"=========================Actor {config.task} epoch {actor_epoch} starts=========================")
            # load ckpt model
            agent.load_model(ckpt_path, evaluate=True)

            # take action to get rp_buffer
            # train model
            # take action to get model_rp_buffer
            model_rp_buffer = maybe_replace_model_rp_buffer()

            # take step_epoch actions
            s0 = env.reset()
            s0_rec_buffer = np.zeros([s_dim])
            s1_rec_buffer = np.zeros([s_dim])
            s0_rec_buffer[-1*params.dict['state_dim']:] = s0
            if params.dict['recurrent']:
                a = agent.get_action(s0_rec_buffer, not config.eval)
            else:
                a = agent.get_action(s0, not config.eval)
            a = a[0][0]
            env.write_action(a)
            fd_list = []

            for steps_epoch in range(epoch_length):
                # take action and add to the env buffer
                s1, r, terminal, error_code = env.step(a, eval_=config.eval)
                if error_code == True:
                    s1_rec_buffer = np.concatenate((s0_rec_buffer[params.dict['state_dim']:], s1) )
                    if params.dict['recurrent']:
                        a1 = agent.get_action(s1_rec_buffer, not config.eval)
                    else:
                        a1 = agent.get_action(s1,not config.eval)
                    a1 = a1[0][0]
                    env.write_action(a1)
                else:
                    print("TaskID:"+str(config.task)+"Invalid state received...\n")
                    env.write_action(a)
                    continue

                if params.dict['recurrent']:
                    fd = (s0_rec_buffer, a, np.array([r]), s1_rec_buffer, np.array([terminal], np.float))
                else:
                    fd = (s0, a, np.array([r]), s1, np.array([terminal], np.float))
                
                if not config.eval:
                    fd_list.append(fd)
                
                s0 = s1
                a = a1
                if params.dict['recurrent']:
                    s0_rec_buffer = s1_rec_buffer

                if not params.dict['use_TCP'] and (terminal):
                    if agent.actor_noise != None:
                        agent.actor_noise.reset()

                if steps_epoch == epoch_length - 1:
                    print(f" ==========================Actor {config.task} starts evaluation===================")
                    eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, config, params, s0_rec_buffer, eval_step_counter, f_log_file)
                    print(f" ==========================Actor {config.task} finishes evaluation===================")
            
            # write the rp_buffer
            actor_rp_f = open(actor_rp_file_path, 'wb')
            write_rp(actor_rp_f, fd_list)
            for data in fd_list:     
                agent.store_experience(data[0], data[1], data[2], data[3], data[4])
                
            # train dynamics model
            agent.train_model_and_save_model_and_data()

            # rollout model and populate the model_rp_buffer
            new_model_buffer_data_list = agent.rollout_model_and_populate_model_rp_buffer(rollout_length)

            # write the model_rp_buffer
            # TODO: distinguish whether append or write, but now for the one actor case, always write
            actor_model_rp_f = open(actor_model_rp_file_path, 'wb')
            write_rp(actor_model_rp_f, new_model_buffer_data_list)

            # write the signal file: the actor has finished the epoch
            signal_f = open(signal_file_path, 'w')
            signal_f.write(f"1")
            signal_f.close()

            print(f"total time for actor-{config.task}: {time.time() - start}")
            f_log_file.write(f"total time for actor-{config.task}: {time.time() - start}\n")
            f_log_file.flush()

            print(f"====== Actor {config.task} one step finishes ======")

            actor_epoch += 1


if __name__ == "__main__":
    core()
