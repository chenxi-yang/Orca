'''
pytorch Orca with MBPO(Orca's AC version)
'''

import threading
import logging
import sys
# from agent import Agent
from agent_pytorch import Agent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


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
                s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )

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


class learner_killer():

    def __init__(self, buffer):

        self.replay_buf = buffer
        print("learner register sigterm")
        signal.signal(signal.SIGTERM, self.handler_term)
        print("test length:", self.replay_buf.length_buf)

    def handler_term(self, signum, frame): # TODO: signum, frame not used
        if not config.eval:
            with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), "wb") as fp:
                pickle.dump(self.replay_buf, fp)
                print("test length:", self.replay_buf.length_buf)
                print("--------------------------Learner: Saving rp memory--------------------------")
        print("-----------------------Learner's killed---------------------")
        sys.exit(0)


def main():

    # TODO update pytorch logger
    # tf.get_logger().setLevel(logging.ERROR)

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
    parser.add_argument('--pytorch_logdir', type=str, default="pytorch_train_dir")
    parser.add_argument('--actor_max_epochs', type=int, default = 200) # per actor # epoch #TODO: should be 50k use 500 for now
    parser.add_argument('--num_ac_updates_per_step', type=int, default = 1) # per actor # step, Orca's default is 1

    # new parameters
    parser.add_argument('--rp_dir', action='store_true', default=f"rp_dir", help='default is  %(default)s')

    ## parameters from parser
    global config
    global params
    config = parser.parse_args()
    
    training_session = config.training_session
    print(f"training_session: {training_session}")
    
    CKPT_DIR = f"./rl-module/pytorch_train_dir"
    CKPT_DIR = CKPT_DIR + "_" + str(training_session)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    finish_file = os.path.join(CKPT_DIR, "finish.txt")

    ## parameters from file
    params = Params(os.path.join(config.base_path, 'params.json'))

    if params.dict['single_actor_eval']: # evaluate the single actor
        def is_actor_fn(i): 
            return True
        is_learner = False
    else:
        is_learner = config.job_name == 'learner'
        def is_actor_fn(i): 
            print(f"is actor fn: {i}, task: {config.task}")
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

    pytorchevent_dir = os.path.join(config.base_path, config.pytorch_logdir, config.job_name+str(config.task))
    params.dict['train_dir'] = pytorchevent_dir

    #TODO: pytorch logger
    if not os.path.exists(pytorchevent_dir):
        os.makedirs(pytorchevent_dir)
        
    summary_writer = None
    agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer, h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],
                    noise_exp=params.dict['noise_exp'], train_dir=CKPT_DIR, mix_env=config.mix_env, num_actors=params.dict['num_actors']) # device='cpu'
    # If mix_env, agent only has one rp buffer. Othervise, agent has multiple rp buffers

    if is_learner:
        log_file_path = f"{CKPT_DIR}/learner_training_log.txt"
        f_log_file = open(log_file_path, "w")
        
        # TODO: no load for now
        # if config.load is True and config.eval==False:
        #     if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
        #         with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
        #             replay_memory = pickle.load(fp)

        # _killsignal = learner_killer(agent.rp_buffer)
    else:
        log_file_path = f"{CKPT_DIR}/actor{config.task}_training_log.txt"
        f_log_file = open(log_file_path, "w")

    # initialize the environment for each actor
    for i in range(params.dict['num_actors']):
        if is_actor_fn(i):
            if params.dict['use_TCP']:
                shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
                print(f"Shared Memory: {config.mem_r}, {config.mem_w} at actor {i}")
                env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, shrmem_r=shrmem_r, shrmem_w=shrmem_w,use_normalizer=params.dict['use_normalizer'])
            else:
                env = GYM_Env_Wrapper(env_str, params)

    if params.dict['ckptdir'] is not None:
        params.dict['ckptdir'] = os.path.join(config.base_path, params.dict['ckptdir'])
        print("## checkpoint dir:", params.dict['ckptdir'])
        isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
        print("## checkpoint exists?:", isckpt)
        if isckpt== False:
            print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
    else:
        params.dict['ckptdir'] = pytorchevent_dir

    # start the learner
    # each learner, periodically read all the files in the directory (when all the actor finishes)
    # first, store learner's NN data
    # and assign the parameters to the actor
    # have a signal file (know if all the actors are finished or not)
    if is_learner: # is learner
        print("=========================Learner is up===================")

        if config.load is False:
            agent.init_target()

        counter = 0
        start = time.time()

        # save model (the randomly initialized model)
        agent.save_model()

        signal_file_path_lists = [os.path.join(CKPT_DIR, "signal_file_" + str(i) + ".txt") for i in range(params.dict['num_actors'])]
        actor_rp_file_path_lists = [os.path.join(RP_DIR, "actor_rp_file_" + str(i) + "_" + str(training_session) + ".txt") for i in range(params.dict['num_actors'])]
        
        # initialize the signal file
        for signal_file_path in signal_file_path_lists:
            with open(signal_file_path, 'w') as f_actor_signal:
                pass
            f_actor_signal.close()

        env_steps = 0
        epochs = 0

        while env_steps < config.learner_num_steps: # params.dict['max_epochs']: # 1m epochs
            # check the signal file
            # if all the actors are finished, then read all the files
            print(f"==== Learner in Epoch {epochs}; Step {env_steps} ====")
            epoch_start_time = time.time()
            finished_actor_list = []

            # Learner: listen to the signal from environment interactions
            while True:
                #read the signal file
                finished_actor_list = []
                actor_i = 0
                for signal_file_path in signal_file_path_lists:
                    if os.path.exists(signal_file_path) is False:
                        break
                    f_actor_signal = open(signal_file_path, 'r')
                    tmp_signal = f_actor_signal.read()
                    if tmp_signal == '1':
                        finished_actor_list.append(actor_i)
                    else:
                        break
                    f_actor_signal.close() 
                    actor_i += 1
                # If there is an actor finishes, jump out and select from all finished actor RBs
                if len(finished_actor_list) > 0:
                    break
            
            # add sac buffer to the learner's buffer
            # read all the available rp files
            rp_idx = 0
            for actor_rp_file_path in actor_rp_file_path_lists:
                # print(f"Loading buffer from {actor_rp_file_path}")
                if rp_idx in finished_actor_list:
                    f_actor_rp = open(actor_rp_file_path, 'rb') # read binary
                    buffer_data = pickle.load(f_actor_rp)
                    for data in buffer_data:     
                        agent.store_experience(data[0], data[1], data[2], data[3], data[4], idx=rp_idx)
                rp_idx += 1

            if config.mix_env:
                print(f"agent's rp length: {agent.rp_buffer.length_buf}")
            else:
                print(f"agent's rp length: {[agent.rp_buffer_list[i].length_buf for i in range(params.dict['num_actors'])]}")
            # finish reading data from rp
            # update a signal file (learner_finish_reading_rp)
            # TODO: current strategy: one random seed for all steps here
            
            if not config.mix_env:
                selected_env = random.choice(finished_actor_list)
            else:
                selected_env = None
            
            # Do the real training step
            rollout_length = 20 # hyperparameter of MBPO
            effective_model_rollouts_per_step = 400
            freq_train_model = 200
            epoch_length = 60 # length of one-time TCP interaction
            num_ac_updates_per_step = 5 # number of updates to the learner
            sac_updates_every_steps = 1
            ac_batch_size = 256

            ac_buffer_capacity = rollout_length * effective_model_rollouts_per_step * freq_train_model
            ac_buffer_capacity *= 1 # hyperparameter
            # TODO: omit for now
            # ac_buffer = maybe_replace_ac_buffer(
            #     ac_buffer,
            #     ac_buffer_capacity,
            # )

            obs, done = None, False
            for steps_epoch in range(epoch_length):
                # I have the buffer now
                # list of models
                if (env_steps + 1) % freq_train_model == 0:
                # train model
                # train_model_save_model_and_data()
                # train the selected model
                # roll_out_model_and_populate_ac_buffer # ac_buffer is the buffer for model rollout data
                for _ in range(num_ac_updates_per_step):
                    which_buffer = ac_buffer # always using the ac_buffer for now
                    if (env_steps + 1) % sac_updates_every_steps != 0 or len(which_buffer) < ac_batch_size:
                        break
                
                    agent.train_step(selected_env)
                    if params.dict['use_hard_target'] == False:
                        # if counter % 5 == 0: #TODO:  update the target function every 5 epochs
                        agent.target_update()
                    else:
                        if counter % params.dict['hard_target'] == 0 :
                            agent.target_update() # hard target update
                            
            # for _ in range(config.num_ac_updates_per_step):
            #     agent.train_step(selected_env)
            #     if params.dict['use_hard_target'] == False:
            #         # if counter % 5 == 0: #TODO:  update the target function every 5 epochs
            #         agent.target_update()
            #     else:
            #         if counter % params.dict['hard_target'] == 0 :
            #             agent.target_update() # hard target update
            print(f"Epoch: {counter}, Loss/learner's actor_loss: {agent.a_loss}, time: {time.time() - epoch_start_time}")
            f_log_file.write(f"Epoch: {counter}, Loss/learner's actor_loss: {agent.a_loss}, time: {time.time() - epoch_start_time}\n")
            f_log_file.flush()
            
            counter += 1

            agent.save_model()
            # clean up the rp file
            # clean up the signal file
            signal_idx = 0
            for signal_file_path in signal_file_path_lists:
                if signal_idx in finished_actor_list:
                    f_actor_signal = open(signal_file_path, 'w')
                    f_actor_signal.close()
                signal_idx += 1
        with open(finish_file, 'w') as f_finish:
            f_finish.write('1')
        f_finish.close()

    else: # is normal actor
        # load NN from learner's file
        # constantly write the replay buffer to the directory
        # start with one actor version
        # store a signal to a signal file (know if all the actors are finished or not) with the actor idx
        print(f"=========================Actor {config.task} is up===================")
        signal_file_path = os.path.join(CKPT_DIR, "signal_file_" + str(config.task) + ".txt")
        actor_rp_file_path = os.path.join(RP_DIR, "actor_rp_file_" + str(config.task) + "_" + str(training_session) + ".txt")
        # ckpt_path = os.path.join(CKPT_DIR, "model")
        ckpt_path = f"{CKPT_DIR}/trained_model/model.pth"
        
        # print(f"current path: {os.getcwd()}")
        # print(f"ckpt path: {ckpt_path}")
        print(f"ckpt exists: {os.path.exists(ckpt_path)}")
        print(f"rp exists: {os.path.exists(actor_rp_file_path)}")
        finish_flag = False
        while True:
            print(f"Wait for the learner to finish.")
            while True:
                # if cp file exists and (actor_rp_file not exists or actor_rp_file is empty) ==> learner is done reading the replay buffer
                # break
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
                    
            print(f"learner finish: {finish_flag}")
            if finish_flag:
                break

            print(f"=========================Actor {config.task} starts rollout===================")
            # load actor's NN from the cp file
            agent.load_model(ckpt_path, evaluate=True)
            actor_rp_f = open(actor_rp_file_path, 'wb') # write binary
            signal_f = open(signal_file_path, 'w')

            start = time.time()
            step_counter = np.int64(0)
            eval_step_counter = np.int64(0)
            s0 = env.reset()
            s0_rec_buffer = np.zeros([s_dim])
            s1_rec_buffer = np.zeros([s_dim])
            s0_rec_buffer[-1*params.dict['state_dim']:] = s0

            # print(f"s_dim: {s_dim}")

            if params.dict['recurrent']:
                a = agent.get_action(s0_rec_buffer, not config.eval)
            else:
                a = agent.get_action(s0, not config.eval)
            a = a[0][0]
            env.write_action(a)
            epoch = 0
            ep_r = 0.0

            # the mahimahi script sets the max steps to 50000
            # I could manually add this for better readability
            fd_list = []
            while epoch < config.actor_max_epochs:
                epoch += 1

                step_counter += 1
                print(f"---- actor {config.task} ---- start env step")
                s1, r, terminal, error_code = env.step(a, eval_=config.eval)
                print(f"---- actor {config.task} ---- finish env step")
                
                print(f"Results of actor {config.task}: {s1}, {r}, {terminal}, {error_code}")
                
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
                    # pass

                s0 = s1
                a = a1
                if params.dict['recurrent']:
                    s0_rec_buffer = s1_rec_buffer

                if not params.dict['use_TCP'] and (terminal):
                    if agent.actor_noise != None:
                        agent.actor_noise.reset()
                
                if config.actor_max_epochs == 200:
                    if (epoch% 199 == 0):
                        print(f" ==========================Actor {config.task} starts evaluation===================")
                        eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, config, params, s0_rec_buffer, eval_step_counter, f_log_file)
                        print(f" ==========================Actor {config.task} finishes evaluation===================")
                elif config.actor_max_epochs == 500:
                    if (epoch% 499 == 0):
                        eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, config, params, s0_rec_buffer, eval_step_counter, f_log_file)
                else:
                    if (epoch% params.dict['eval_frequency'] == 0):
                # if (epoch % 499 == 0):
                # if (epoch % (params.dict['eval_frequency'] - 1) == 0):
                    # update the log part (for now, print the score)
                        eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, config, params, s0_rec_buffer, eval_step_counter, f_log_file)

                # TODO: when the epoch is enlarged
                # if epoch % 500 == 0:
                # print(f"epoch {epoch} for actor{config.task} finished.")
                
            write_rp(actor_rp_f, fd_list)
            print(f"total time for actor-{config.task}: {time.time() - start}")
            f_log_file.write(f"total time for actor-{config.task}: {time.time() - start}\n")
            f_log_file.flush()
            # write to signal file

            actor_rp_f.close()
            signal_f.write(f"1") # represents that the actor is done
            signal_f.close()
            
            print(f"====== Actor {config.task} one step finishes ======")
        
        

if __name__ == "__main__":
    main()
