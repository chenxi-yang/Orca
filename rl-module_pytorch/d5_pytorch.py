'''
PyTorch version of the d5.py in Orca
'''

import threading
import logging
import tensorflow as tf

import torch
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

            ep_r = ep_r+r


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

    return eval_step_counter


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


    ## parameters from parser
    global config
    global params
    config = parser.parse_args()

    ## parameters from file
    params = Params(os.path.join(config.base_path,'params.json'))

    # TODO: update the distributed training process 
    if params.dict['single_actor_eval']:
        local_job_device = ''
        shared_job_device = ''
        def is_actor_fn(i): return True
        global_variable_device = '/cpu'
        is_learner = False
        # TODO: distributed training
        server = tf.train.Server.create_local_server()
        filters = []
    else:

        local_job_device = '/job:%s/task:%d' % (config.job_name, config.task)
        shared_job_device = '/job:learner/task:0'

        is_learner = config.job_name == 'learner'

        global_variable_device = shared_job_device + '/cpu'


        def is_actor_fn(i): return config.job_name == 'actor' and i == config.task

        # TODO: server info
        if params.dict['remote']:
            cluster = tf.train.ClusterSpec({
                'actor': params.dict['actor_ip'][:params.dict['num_actors']],
                'learner': [params.dict['learner_ip']]
            })
        else:
            cluster = tf.train.ClusterSpec({
                    'actor': ['localhost:%d' % (8001 + i) for i in range(params.dict['num_actors'])],
                    'learner': ['localhost:8000']
                })

        # TODO: train on server clusters
        server = tf.train.Server(cluster, job_name=config.job_name,
                                task_index=config.task)
        filters = [shared_job_device, local_job_device]


    # set up the TCP usage
    if params.dict['use_TCP']:
        env_str = "TCP"
        env_peek = TCP_Env_Wrapper(env_str, params,use_normalizer=params.dict['use_normalizer'])

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


    # TODO: use tf graph, for training
    # todo: create TF graph
    # set up the default graph and device
    with tf.Graph().as_default(),\
        tf.device(local_job_device + '/cpu'):
        # pytorch set cuda

        # torch.random.manual_seed(1234)
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        actor_op = []
        now = datetime.datetime.now()
        tfeventdir = os.path.join(config.base_path, params.dict['logdir'], config.job_name+str(config.task) )
        params.dict['train_dir'] = tfeventdir

        if not os.path.exists(tfeventdir):
            os.makedirs(tfeventdir)
        
        # TODO: tf.summary?
        summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

        # TODO: with pytorch device
        # with torch.device(shared_job_device):
        with tf.device(shared_job_device):
            
            # TODO: update the summary, ini agent
            # TODO: update the device setting
            agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                        h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                        lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                        LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],
                        noise_exp=params.dict['noise_exp'], device=params.dict['device'])

            # todo: pytorch dtypes
            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]

            # TODO: torch.FIFOQueue?
            # should use a normal tensor Queue for pytorch
            queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")


        if is_learner:
            # TODO: pytorch device
            with tf.device(params.dict['device']):
                agent.build_learn()
                # todo: summary/log
                agent.create_tf_summary()

            if config.load is True and config.eval==False:
                if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                    with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                        replay_memory = pickle.load(fp)

            # TODO: check learner_killer
            _killsignal = learner_killer(agent.rp_buffer)


        for i in range(params.dict['num_actors']):
                if is_actor_fn(i):
                    if params.dict['use_TCP']:
                        shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                        shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
                        env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, shrmem_r=shrmem_r, shrmem_w=shrmem_w,use_normalizer=params.dict['use_normalizer'])
                    else:
                        env = GYM_Env_Wrapper(env_str, params)

                    # TODO: no need for pytorch??
                    a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0') # throughput, delay, loss_rate
                    a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action') # cwnd, or f(cwnd)
                    a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward') #
                    a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1')
                    a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal')
                    a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal] # s, a, r, s', done

                    # TODO: for each actor, create a thread
                    # add a buffer to queue
                    with tf.device(shared_job_device):
                        actor_op.append(queue.enqueue(a_buf))

        if is_learner:
            Dequeue_Length = params.dict['dequeue_length']
            dequeue = queue.dequeue_many(Dequeue_Length)

        queuesize_op = queue.size()

        if params.dict['ckptdir'] is not None:
            params.dict['ckptdir'] = os.path.join( config.base_path, params.dict['ckptdir'])
            print("## checkpoint dir:", params.dict['ckptdir'])
            isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
            print("## checkpoint exists?:", isckpt)
            if isckpt== False:
                print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
        else:
            params.dict['ckptdir'] = tfeventdir

        # TODO: tf config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        # TODO: tf, set up session
        if params.dict['single_actor_eval']:
            mon_sess = tf.train.SingularMonitoredSession(
                checkpoint_dir=params.dict['ckptdir'])
        else:
            # TODO: pytorch distributed training
            mon_sess = tf.train.MonitoredTrainingSession(master=server.target,
                    save_checkpoint_secs=15,
                    save_summaries_secs=None,
                    save_summaries_steps=None,
                    is_chief=is_learner, # the learner is the chief
                    checkpoint_dir=params.dict['ckptdir'],
                    config=tfconfig, # TODO: tf link, e.g. localhost:2222
                    hooks=None)

        # TODO: assign session to agent
        agent.assign_sess(mon_sess)


        if is_learner: # is learner

            if config.eval is True:
                print("=========================Learner is up===================")
                while not mon_sess.should_stop():
                    time.sleep(1)
                    continue

            if config.load is False:
                # TODO: pytorch, init target session
                agent.init_target()

            counter = 0
            start = time.time()

            dequeue_thread = threading.Thread(target=learner_dequeue_thread, args=(agent,params, mon_sess, dequeue, queuesize_op, Dequeue_Length),daemon=True)
            first_time=True

            while not mon_sess.should_stop():

                if first_time == True:
                    dequeue_thread.start()
                    first_time=False

                up_del_tmp=params.dict['update_delay']/1000.0
                time.sleep(up_del_tmp)
                if agent.rp_buffer.ptr>200 or agent.rp_buffer.full :
                    agent.train_step()
                    if params.dict['use_hard_target'] == False:
                        # TODO: pytorch update target
                        agent.target_update()

                        if counter %params.dict['hard_target'] == 0 :
                            # TODO: run session/agent, what is global_step?
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))

                    else:
                        if counter %params.dict['hard_target'] == 0 :

                            agent.target_update()
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))

                counter += 1


        else: # is normal actor
                start = time.time()
                step_counter = np.int64(0)
                eval_step_counter = np.int64(0)
                s0 = env.reset()
                s0_rec_buffer = np.zeros([s_dim])
                s1_rec_buffer = np.zeros([s_dim])
                s0_rec_buffer[-1*params.dict['state_dim']:] = s0


                if params.dict['recurrent']:
                    a = agent.get_action(s0_rec_buffer,not config.eval)
                else:
                    a = agent.get_action(s0, not config.eval)
                a = a[0][0]
                env.write_action(a)
                epoch = 0
                ep_r = 0.0
                start = time.time()
                while True:
                    start = time.time()
                    epoch += 1

                    step_counter += 1
                    s1, r, terminal, error_code = env.step(a,eval_=config.eval)

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
                        fd = {a_s0:s0_rec_buffer, a_action:a, a_reward:np.array([r]), a_s1:s1_rec_buffer, a_terminal:np.array([terminal], np.float)}
                    else:
                        fd = {a_s0:s0, a_action:a, a_reward:np.array([r]), a_s1:s1, a_terminal:np.array([terminal], np.float)}

                    if not config.eval:
                        # TODO: update the running part
                        # distribute over actors
                        mon_sess.run(actor_op, feed_dict=fd)

                    s0 = s1
                    a = a1
                    if params.dict['recurrent']:
                        s0_rec_buffer = s1_rec_buffer

                    if not params.dict['use_TCP'] and (terminal):
                        if agent.actor_noise != None:
                            agent.actor_noise.reset()

                    if (epoch% params.dict['eval_frequency'] == 0):
                        eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer, params, s0_rec_buffer, eval_step_counter)

                print("total time:", time.time()-start)

def learner_dequeue_thread(agent, params, mon_sess, dequeue, queuesize_op, Dequeue_Length):
    ct = 0
    while True:
        ct = ct + 1
        # TODO: what is mon_sess's input, how did it change the queue?
        data = mon_sess.run(dequeue)
        agent.store_many_experience(data[0], data[1], data[2], data[3], data[4], Dequeue_Length)
        time.sleep(0.01)


def learner_update_thread(agent,params):
    delay=params.dict['update_delay']/1000.0
    ct = 0
    while True:
        agent.train_step()
        agent.target_update()
        time.sleep(delay)


if __name__ == "__main__":
    main()
