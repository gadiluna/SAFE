# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/

import argparse
import time
import sys, os
import logging


#
# Parameters File for the SAFE network.
#
# Authors: SAFE team


def getLogger(logfile):
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr


class Flags:

    def __init__(self):
        parser = argparse.ArgumentParser(description='SAFE')

        parser.add_argument("-o", "--output", dest="output_file", help="output directory for logging and models",
                            required=False)
        parser.add_argument("-e", "--embedder", dest="embedder_folder",
                            help="file with the embedding matrix and dictionary for asm instructions", required=False)
        parser.add_argument("-n", "--dbName", dest="db_name", help="Name of the database", required=False)
        parser.add_argument("-ld", "--load_dir", dest="load_dir", help="Load the model from directory load_dir",
                            required=False)
        parser.add_argument("-r", "--random", help="if present the network use random embedder", default=False,
                            action="store_true", dest="random_embedding", required=False)
        parser.add_argument("-te", "--trainable_embedding",
                            help="if present the network consider the embedding as trainable", action="store_true",
                            dest="trainable_embeddings", default=False)
        parser.add_argument("-cv", "--cross_val", help="if present the training is done with cross validiation",
                            default=False, action="store_true", dest="cross_val")

        args = parser.parse_args()

        # mode = mean_field
        self.batch_size = 250  # minibatch size (-1 = whole dataset)
        self.num_epochs = 50  # number of epochs
        self.embedding_size = 100  # dimension of the function embedding
        self.learning_rate = 0.001  # init learning_rate
        self.l2_reg_lambda = 0  # 0.002 #0.002 # regularization coefficient
        self.num_checkpoints = 1  # max number of checkpoints
        self.out_dir = args.output_file  # directory for logging
        self.rnn_state_size = 50  # dimesion of the rnn state
        self.db_name = args.db_name
        self.load_dir = str(args.load_dir)
        self.random_embedding = args.random_embedding
        self.trainable_embeddings = args.trainable_embeddings
        self.cross_val = args.cross_val
        self.cross_val_fold = 5

        #
        ##
        ## RNN PARAMETERS, these parameters are only used for RNN model.
        #
        self.rnn_depth = 1  # depth of the rnn
        self.max_instructions = 150  # number of instructions

        ## ATTENTION PARAMETERS
        self.attention_hops = 10
        self.attention_depth = 250

        # RNN SINGLE PARAMETER
        self.dense_layer_size = 2000

        self.seed = 2  # random seed

        # create logdir and logger
        self.reset_logdir()

        self.embedder_folder = args.embedder_folder

    def reset_logdir(self):
        # create logdir
        timestamp = str(int(time.time()))
        self.logdir = os.path.abspath(os.path.join(self.out_dir, "runs", timestamp))
        os.makedirs(self.logdir, exist_ok=True)

        # create logger
        self.log_file = str(self.logdir) + '/console.log'
        self.logger, self.hdlr = getLogger(self.log_file)

        # create symlink for last_run
        sym_path_logdir = str(self.out_dir) + "/last_run"
        try:
            os.unlink(sym_path_logdir)
        except:
            pass
        try:
            os.symlink(self.logdir, sym_path_logdir)
        except:
            print("\nfailed to create symlink!\n")

    def close_log(self):
        self.hdlr.close()
        self.logger.removeHandler(self.hdlr)
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __str__(self):
        msg = ""
        msg += "\nParameters:\n"
        msg += "\tRandom embedding: {}\n".format(self.random_embedding)
        msg += "\tTrainable embedding: {}\n".format(self.trainable_embeddings)
        msg += "\tlogdir: {}\n".format(self.logdir)
        msg += "\tbatch_size: {}\n".format(self.batch_size)
        msg += "\tnum_epochs: {}\n".format(self.num_epochs)
        msg += "\tembedding_size: {}\n".format(self.embedding_size)
        msg += "\trnn_state_size: {}\n".format(self.rnn_state_size)
        msg += "\tattention depth: {}\n".format(self.attention_depth)
        msg += "\tattention hops: {}\n".format(self.attention_hops)
        msg += "\tdense layer e: {}\n".format(self.dense_layer_size)

        msg += "\tlearning_rate: {}\n".format(self.learning_rate)
        msg += "\tl2_reg_lambda: {}\n".format(self.l2_reg_lambda)
        msg += "\tnum_checkpoints: {}\n".format(self.num_checkpoints)


        msg += "\tseed: {}\n".format(self.seed)
        msg += "\tMax Instructions per functions: {}\n".format(self.max_instructions)
        return msg
