# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt)
#

from SiameseSAFE import SiameseSelfAttentive
from PairFactory import PairFactory
import tensorflow as tf
import random
import sys, os
import numpy as np
from sklearn import metrics
import matplotlib
import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class modelSAFE:

    def __init__(self, flags, embedding_matrix):
        self.embedding_size = flags.embedding_size
        self.num_epochs = flags.num_epochs
        self.learning_rate = flags.learning_rate
        self.l2_reg_lambda = flags.l2_reg_lambda
        self.num_checkpoints = flags.num_checkpoints
        self.logdir = flags.logdir
        self.logger = flags.logger
        self.seed = flags.seed
        self.batch_size = flags.batch_size
        self.max_instructions = flags.max_instructions
        self.embeddings_matrix = embedding_matrix
        self.session = None
        self.db_name = flags.db_name
        self.trainable_embeddings = flags.trainable_embeddings
        self.cross_val = flags.cross_val
        self.attention_hops = flags.attention_hops
        self.attention_depth = flags.attention_depth
        self.dense_layer_size = flags.dense_layer_size
        self.rnn_state_size = flags.rnn_state_size

        random.seed(self.seed)
        np.random.seed(self.seed)

        print(self.db_name)

    # loads an usable model
    # returns the network and a tensorflow session in which the network can be used.
    @staticmethod
    def load_model(path):
        session = tf.Session()
        checkpoint_dir = os.path.abspath(os.path.join(path, "checkpoints"))
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, "model.meta"))
        tf.global_variables_initializer().run(session=session)
        saver.restore(session, os.path.join(checkpoint_dir, "model"))
        network = SiameseSelfAttentive(
            rnn_state_size=1,
            learning_rate=1,
            l2_reg_lambda=1,
            batch_size=1,
            max_instructions=1,
            embedding_matrix=1,
            trainable_embeddings=1,
            attention_hops=1,
            attention_depth=1,
            dense_layer_size=1,
            embedding_size=1
        )
        network.restore_model(session)
        return session, network

    def create_network(self):
        self.network = SiameseSelfAttentive(
            rnn_state_size=self.rnn_state_size,
            learning_rate=self.learning_rate,
            l2_reg_lambda=self.l2_reg_lambda,
            batch_size=self.batch_size,
            max_instructions=self.max_instructions,
            embedding_matrix=self.embeddings_matrix,
            trainable_embeddings=self.trainable_embeddings,
            attention_hops=self.attention_hops,
            attention_depth=self.attention_depth,
            dense_layer_size=self.dense_layer_size,
            embedding_size=self.embedding_size
        )

    def train(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )
            sess = tf.Session(config=session_conf)

            # Sets the graph-level random seed.
            tf.set_random_seed(self.seed)

            self.create_network()
            self.network.generate_new_safe()
            # --tbrtr

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # TensorBoard
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", self.network.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary])
            train_summary_dir = os.path.join(self.logdir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            val_summary_op = tf.summary.merge([loss_summary])
            val_summary_dir = os.path.join(self.logdir, "summaries", "validation")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary])
            test_summary_dir = os.path.join(self.logdir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(self.logdir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

            best_val_auc = 0
            stat_file = open(str(self.logdir) + "/epoch_stats.tsv", "w")
            stat_file.write("#epoch\ttrain_loss\tval_loss\tval_auc\ttest_loss\ttest_auc\n")

            p_train = PairFactory(self.db_name, 'train_pairs', self.batch_size, self.max_instructions)
            p_validation = PairFactory(self.db_name, 'validation_pairs', self.batch_size, self.max_instructions, False)
            p_test = PairFactory(self.db_name, 'test_pairs', self.batch_size, self.max_instructions, False)

            step = 0
            for epoch in range(0, self.num_epochs):
                epoch_msg = ""
                epoch_msg += "  epoch: {}\n".format(epoch)

                epoch_loss = 0

                # ----------------------#
                #         TRAIN	       #
                # ----------------------#
                n_batch = 0
                for function1_batch, function2_batch, len1_batch, len2_batch, y_batch in tqdm.tqdm(
                        p_train.async_chunker(epoch % 25), total=p_train.num_batches):
                    feed_dict = {
                        self.network.x_1: function1_batch,
                        self.network.x_2: function2_batch,
                        self.network.lengths_1: len1_batch,
                        self.network.lengths_2: len2_batch,
                        self.network.y: y_batch,
                    }

                    summaries, _, loss, norms, cs = sess.run(
                        [train_summary_op, self.network.train_step, self.network.loss, self.network.norms,
                         self.network.cos_similarity],
                        feed_dict=feed_dict)

                    train_summary_writer.add_summary(summaries, step)
                    epoch_loss += loss * p_train.batch_dim  # ???
                    step += 1
                # recap epoch
                epoch_loss /= p_train.num_pairs
                epoch_msg += "\ttrain_loss: {}\n".format(epoch_loss)

                # ----------------------#
                #      VALIDATION	   #
                # ----------------------#
                val_loss = 0
                epoch_msg += "\n"
                val_y = []
                val_pred = []
                for function1_batch, function2_batch, len1_batch, len2_batch, y_batch in tqdm.tqdm(
                        p_validation.async_chunker(0), total=p_validation.num_batches):
                    feed_dict = {
                        self.network.x_1: function1_batch,
                        self.network.x_2: function2_batch,
                        self.network.lengths_1: len1_batch,
                        self.network.lengths_2: len2_batch,
                        self.network.y: y_batch,
                    }

                    summaries, loss, similarities = sess.run(
                        [val_summary_op, self.network.loss, self.network.cos_similarity], feed_dict=feed_dict)
                    val_loss += loss * p_validation.batch_dim
                    val_summary_writer.add_summary(summaries, step)
                    val_y.extend(y_batch)
                    val_pred.extend(similarities.tolist())

                val_loss /= p_validation.num_pairs

                if np.isnan(val_pred).any():
                    print("Validation: carefull there is  NaN in some ouput values, I am fixing it but be aware...")
                    val_pred = np.nan_to_num(val_pred)

                val_fpr, val_tpr, val_thresholds = metrics.roc_curve(val_y, val_pred, pos_label=1)
                val_auc = metrics.auc(val_fpr, val_tpr)
                epoch_msg += "\tval_loss : {}\n\tval_auc : {}\n".format(val_loss, val_auc)

                sys.stdout.write(
                    "\r\tepoch {} / {}, loss {:g}, val_auc {:g}, norms {}".format(epoch, self.num_epochs, epoch_loss,
                                                                                  val_auc, norms))
                sys.stdout.flush()

                # execute test only if validation auc increased
                test_loss = "-"
                test_auc = "-"

                # in case of cross validation we do not need to evaluate on a test split that is effectively missing
                if val_auc > best_val_auc and self.cross_val:
                    #
                    ##--  --##
                    #
                    best_val_auc = val_auc
                    saver.save(sess, checkpoint_prefix)
                    print("\nNEW BEST_VAL_AUC: {} !\n".format(best_val_auc))
                    # write ROC raw data
                    with open(str(self.logdir) + "/best_val_roc.tsv", "w") as the_file:
                        the_file.write("#thresholds\ttpr\tfpr\n")
                        for t, tpr, fpr in zip(val_thresholds, val_tpr, val_fpr):
                            the_file.write("{}\t{}\t{}\n".format(t, tpr, fpr))

                # in case we are not cross validating we expect to have a test split.
                if val_auc > best_val_auc and not self.cross_val:

                    best_val_auc = val_auc
                    epoch_msg += "\tNEW BEST_VAL_AUC: {} !\n".format(best_val_auc)

                    # save best model
                    saver.save(sess, checkpoint_prefix)

                    # ----------------------#
                    #         TEST  	    #
                    # ----------------------#

                    # TEST
                    test_loss = 0
                    epoch_msg += "\n"
                    test_y = []
                    test_pred = []

                    for function1_batch, function2_batch, len1_batch, len2_batch, y_batch in tqdm.tqdm(
                            p_test.async_chunker(0), total=p_test.num_batches):
                        feed_dict = {
                            self.network.x_1: function1_batch,
                            self.network.x_2: function2_batch,
                            self.network.lengths_1: len1_batch,
                            self.network.lengths_2: len2_batch,
                            self.network.y: y_batch,
                        }
                        summaries, loss, similarities = sess.run(
                            [test_summary_op, self.network.loss, self.network.cos_similarity], feed_dict=feed_dict)
                        test_loss += loss * p_test.batch_dim
                        test_summary_writer.add_summary(summaries, step)
                        test_y.extend(y_batch)
                        test_pred.extend(similarities.tolist())

                    test_loss /= p_test.num_pairs
                    if np.isnan(test_pred).any():
                        print("Test: carefull there is  NaN in some ouput values, I am fixing it but be aware...")
                        test_pred = np.nan_to_num(test_pred)

                    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(test_y, test_pred, pos_label=1)

                    # write ROC raw data
                    with open(str(self.logdir) + "/best_test_roc.tsv", "w") as the_file:
                        the_file.write("#thresholds\ttpr\tfpr\n")
                        for t, tpr, fpr in zip(test_thresholds, test_tpr, test_fpr):
                            the_file.write("{}\t{}\t{}\n".format(t, tpr, fpr))

                    test_auc = metrics.auc(test_fpr, test_tpr)
                    epoch_msg += "\ttest_loss : {}\n\ttest_auc : {}\n".format(test_loss, test_auc)
                    fig = plt.figure()
                    plt.title('Receiver Operating Characteristic')
                    plt.plot(test_fpr, test_tpr, 'b',
                             label='AUC = %0.2f' % test_auc)
                    fig.savefig(str(self.logdir) + "/best_test_roc.png")
                    print(
                        "\nNEW BEST_VAL_AUC: {} !\n\ttest_loss : {}\n\ttest_auc : {}\n".format(best_val_auc, test_loss,
                                                                                               test_auc))
                    plt.close(fig)

                stat_file.write(
                    "{}\t{}\t{}\t{}\t{}\t{}\n".format(epoch, epoch_loss, val_loss, val_auc, test_loss, test_auc))
                self.logger.info("\n{}\n".format(epoch_msg))
            stat_file.close()
            sess.close()
            return best_val_auc
