'''
PyTorch version of the d5.py in Orca
'''

import threading
import logging
# import tensorflow as tf

import torch
import torch.distributed as dist
import sys
# from agent import Agent
from agent_pytorch import Agent
import os
# TODO set up new logger path
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

# noise type: default: gaussian
# run learner and actor separately

# run single actor first and then add the distributed training
# use a shared memory to store the replay buffer
# # use a shared queue/shared directory
# # use a file to store the replay buffer
# each learner, periodically read all the files in the directory (when all the actor finishes)
 # first, store learner's NN data
# and assign the parameters to the actor
# each actor, 
# load NN from learner's file
# constantly write the replay buffer to the directory
# start with one actor version
# have a signal file (know if all the actors are finished or not)

CKPT_DIR = "./pytorch_train_dir"
RP_DIR = "./rp_dir"

# GLOBAL DATA DIRECTORY
def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)

def evaluate_TCP(env, agent, epoch, summary_writer, params, s0_rec_buffer, eval_step_counter):
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

            # TODO: add pytorch logger
            # if (step_counter+1) % params.dict['tb_interval'] == 0:

            #     summary = tf.summary.Summary()
            #     summary.value.add(tag='Eval/Step/0-Actions', simple_value=env.map_action(a))
            #     summary.value.add(tag='Eval/Step/2-Reward', simple_value=r)
            # summary_writer.add_summary(summary, eval_step_counter)

            s0 = s1
            a = a1
            if params.dict['recurrent']:
                s0_rec_buffer = s1_rec_buffer

            if step_counter == eval_length or terminal:
                score_list.append(ep_r)
                break

    # TODO: add pytorch logger
    # summary = tf.summary.Summary()
    # summary.value.add(tag='Eval/Return', simple_value=np.mean(score_list))
    # summary_writer.add_summary(summary, epoch)
    print(f"Eval/Return of actor {params.dict['task']}: {np.mean(score_list)}")

    return eval_step_counter


def write_rp(f, fd): # actor writes one line of replay buffer to the file
    for key, val in fd.items():
        if isinstance(val, np.ndarray):
            for i in range(len(val)):
                if i == len(val) - 1:
                    f.write(str(val[i]))
                else:
                    f.write(str(val[i]) + ",")
        else:
            f.write(str(val))
        f.write(";")
    f.write("\n")
    f.flush()


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
    parser.add_argument('--task', type=int, required=True, help='Task id')
    parser.add_argument('--pytorch_logdir', type=str, default="pytorch_train_dir")
    parser.add_argument('--actor_max_epochs', type=int, default = 50000) # per actor # epoch

    # new parameters
    parser.add_argument('--rp_dir', action='store_true', default=f"rp_dir", help='default is  %(default)s')

    ## parameters from parser
    global config
    global params
    config = parser.parse_args()

    ## parameters from file
    params = Params(os.path.join(config.base_path, 'params.json'))

    if params.dict['single_actor_eval']: # evaluate the single actor
        def is_actor_fn(i): 
            return True
        is_learner = False
    else:
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

    pytorchevent_dir = os.path.join(config.base_path, config.pytorch_logdir, config.job_name+str(config.task) )
    params.dict['train_dir'] = pytorchevent_dir

    #TODO: pytorch logger
    if not os.path.exists(pytorchevent_dir):
        os.makedirs(pytorchevent_dir)
        

    summary_writer = None
    agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer, h1_shape=params.dict['h1_shape'],
                    h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                    lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                    LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],
                    noise_exp=params.dict['noise_exp'], device=params.dict['device'])

    if is_learner:
        if config.load is True and config.eval==False:
            if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                    replay_memory = pickle.load(fp)

        _killsignal = learner_killer(agent.rp_buffer)

    # initialize the environment for each actor
    for i in range(params.dict['num_actors']):
        if is_actor_fn(i):
            if params.dict['use_TCP']:
                shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
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
        if config.eval is True:
            print("=========================Learner is up===================")
            # TODO: pytorch, no kill signal
            # while not mon_sess.should_stop():
                # time.sleep(1)
                # continue

        if config.load is False:
            agent.init_target()

        counter = 0
        start = time.time()

        # save model (the randomly initialized model)
        agent.save_model()

        signal_file_path_lists = [os.path.join(CKPT_DIR, "signal_file_" + str(i) + ".txt") for i in range(params.dict['num_actors'])]
        actor_rp_file_path_lists = [os.path.join(RP_DIR, "actor_rp_file_" + str(i) + ".txt") for i in range(params.dict['num_actors'])]

        while counter < params.dict['max_epochs']: # 1m epochs
            # check the signal file
            # if all the actors are finished, then read all the files
            while True:
                #read the signal file
                finish_counter = 0
                for signal_file_path in signal_file_path_lists:
                    f_actor_signal = open(signal_file_path, 'r')
                    finish_counter += int(f_actor_signal.read())
                    f_actor_signal.close() 
                if finish_counter == params.dict['num_actors']: # all the actors are finished
                    break

            # read all the rp files
            for actor_rp_file_path in actor_rp_file_path_lists:
                f_actor_rp = open(actor_rp_file_path, 'r')
                data = []
                for line in f_actor_rp:
                    read_data = line[:-1].split(";")
                    for idx, val in enumerate(read_data):
                        if idx == 2: # reward
                            data.append(np.array([float(val)]))
                        elif ',' not in val:
                            data.append(float(val))
                        else:
                            tmp_val = np.array([float(i) for i in val.split(",")])
                            data.append(tmp_val.astype(np.float))
                    agent.store_experience(data[0], data[1], data[2], data[3], data[4])

            # finish reading data from rp
            # update a signal file (learner_finish_reading_rp)
            agent.train_step()
            if params.dict['use_hard_target'] == False:
                agent.target_update()
            else:
                if counter % params.dict['hard_target'] == 0 :
                    agent.target_update() # hard target update
            print(f"Epoch: {counter}, Loss/learner's actor_loss: {agent.a_loss}")
            counter += 1

            agent.save_model()
            # clean up the rp file
            # clean up the signal file
            for actor_rp_file_path in actor_rp_file_path_lists:
                f_actor_rp = open(actor_rp_file_path, 'w')
                f_actor_rp.close()
            for signal_file_path in signal_file_path_lists:
                f_actor_signal = open(signal_file_path, 'w')
                f_actor_signal.close()

    else: # is normal actor
        # load NN from learner's file
        # constantly write the replay buffer to the directory
        # start with one actor version
        # store a signal to a signal file (know if all the actors are finished or not) with the actor idx
        signal_file_path = os.path.join(CKPT_DIR, "signal_file_" + str(config.task) + ".txt")
        actor_rp_file_path = os.path.join(RP_DIR, "actor_rp_file_" + str(config.task) + ".txt")
        ckpt_path = os.path.join(CKPT_DIR, "model")

        while True:
            # if cp file exists and (actor_rp_file not exists or actor_rp_file is empty) ==> learner is done reading the replay buffer
            # break
            if os.exists(ckpt_path) and (not os.exists(actor_rp_file_path) or os.stat(actor_rp_file_path).st_size == 0):
                break
        
        # load actor's NN from the cp file
        agent.load_model(ckpt_path, evaluate=True)
        actor_rp_f = open(actor_rp_file_path, 'w')
        signal_f = open(signal_file_path, 'w')

        start = time.time()
        step_counter = np.int64(0)
        eval_step_counter = np.int64(0)
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
        epoch = 0
        ep_r = 0.0
        start = time.time()

        # the mahimahi script sets the max steps to 50000
        # I could manually add this for better readability
        while epoch < config.actor_max_epochs:
            epoch += 1

            step_counter += 1
            s1, r, terminal, error_code = env.step(a, eval_=config.eval)

            if error_code == True:
                s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )

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
                fd = {
                    's0': s0_rec_buffer,
                    'a': a,
                    'r': np.array([r]),
                    's1': s1_rec_buffer,
                    'terminal': np.array([terminal], np.float)
                }
            else:
                fd = {
                    's0': s0,
                    'a': a,
                    'r': np.array([r]),
                    's1': s1,
                    'terminal': np.array([terminal], np.float)
                }

            if not config.eval:
                write_rp(actor_rp_f, fd)

            s0 = s1
            a = a1
            if params.dict['recurrent']:
                s0_rec_buffer = s1_rec_buffer

            if not params.dict['use_TCP'] and (terminal):
                if agent.actor_noise != None:
                    agent.actor_noise.reset()

            if (epoch% params.dict['eval_frequency'] == 0):
                # update the log part (for now, print the score)
                eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, params, s0_rec_buffer, eval_step_counter)

        print(f"total time for actor-{config.task}: {time.time() - start}")
        # write to signal file

        actor_rp_f.close()
        signal_f.write(f"1") # represents that the actor is done
        signal_f.close()
        

if __name__ == "__main__":
    main()
