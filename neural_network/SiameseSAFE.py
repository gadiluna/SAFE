import tensorflow as tf
# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt)
#

# Siamese Self-Attentive Network for Binary Similarity:
#
# arXiv Nostro.
#
# based on the self attentive network:arXiv:1703.03130  Z. Lin at al. “A structured self-attentive sentence embedding''
#
# Authors: SAFE team

class SiameseSelfAttentive:

    def __init__(self,
                 rnn_state_size,  # Dimension of the RNN State
                 learning_rate,  # Learning rate
                 l2_reg_lambda,
                 batch_size,
                 max_instructions,
                 embedding_matrix,  # Matrix containg the embeddings for each asm instruction
                 trainable_embeddings,
                 # if this value is True, the embeddings of the asm instruction are modified by the training.
                 attention_hops,  # attention hops parameter r of [1]
                 attention_depth,  # attention detph parameter d_a of [1]
                 dense_layer_size,  # parameter e of [1]
                 embedding_size,  # size of the final function embedding, in our test this is twice the rnn_state_size
                 ):
        self.rnn_depth = 1  # if this value is modified then the RNN becames a multilayer network. In our tests we fix it to 1 feel free to be adventurous.
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.rnn_state_size = rnn_state_size
        self.batch_size = batch_size
        self.max_instructions = max_instructions
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.attention_hops = attention_hops
        self.attention_depth = attention_depth
        self.dense_layer_size = dense_layer_size
        self.embedding_size = embedding_size

        # self.generate_new_safe()

    def restore_model(self, old_session):
        graph = old_session.graph

        self.x_1 = graph.get_tensor_by_name("x_1:0")
        self.x_2 = graph.get_tensor_by_name("x_2:0")
        self.len_1 = graph.get_tensor_by_name("lengths_1:0")
        self.len_2 = graph.get_tensor_by_name("lengths_2:0")
        self.y = graph.get_tensor_by_name('y_:0')
        self.cos_similarity = graph.get_tensor_by_name("siamese_layer/cosSimilarity:0")
        self.loss = graph.get_tensor_by_name("Loss/loss:0")
        self.train_step = graph.get_operation_by_name("Train_Step/Adam")

        return

    def self_attentive_network(self, input_x, lengths):
        # each functions is a list of embeddings id (an id is an index in the embedding matrix)
        # with this we transform it in a list of embeddings vectors.
        embbedded_functions = tf.nn.embedding_lookup(self.instructions_embeddings_t, input_x)

        # We create the GRU RNN
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, embbedded_functions,
                                                                    sequence_length=lengths, dtype=tf.float32,
                                                                    time_major=False)

        # We create the matrix H
        H = tf.concat([output_fw, output_bw], axis=2)

        # We do a tile to account for training batches
        ws1_tiled = tf.tile(tf.expand_dims(self.WS1, 0), [tf.shape(H)[0], 1, 1], name="WS1_tiled")
        ws2_tile = tf.tile(tf.expand_dims(self.WS2, 0), [tf.shape(H)[0], 1, 1], name="WS2_tiled")

        # we compute the matrix A
        self.A = tf.nn.softmax(tf.matmul(ws2_tile, tf.nn.tanh(tf.matmul(ws1_tiled, tf.transpose(H, perm=[0, 2, 1])))),
                               name="Attention_Matrix")
        # embedding matrix M
        M = tf.identity(tf.matmul(self.A, H), name="Attention_Embedding")

        # we create the flattened version of M
        flattened_M = tf.reshape(M, [tf.shape(M)[0], self.attention_hops * self.rnn_state_size * 2])

        return flattened_M

    def generate_new_safe(self):
        self.instructions_embeddings_t = tf.Variable(initial_value=tf.constant(self.embedding_matrix),
                                                     trainable=self.trainable_embeddings,
                                                     name="instructions_embeddings", dtype=tf.float32)

        self.x_1 = tf.placeholder(tf.int32, [None, self.max_instructions],
                                  name="x_1")  # List of instructions for Function 1
        self.lengths_1 = tf.placeholder(tf.int32, [None], name='lengths_1')  # List of lengths for Function 1
        # example  x_1=[[mov,add,padding,padding],[mov,mov,mov,padding]]
        # lenghts_1=[2,3]

        self.x_2 = tf.placeholder(tf.int32, [None, self.max_instructions],
                                  name="x_2")  # List of instructions for Function 2
        self.lengths_2 = tf.placeholder(tf.int32, [None], name='lengths_2')  # List of lengths for Function 2
        self.y = tf.placeholder(tf.float32, [None], name='y_')  # Real label of the pairs, +1 similar, -1 dissimilar.

        # Euclidean norms; p = 2
        self.norms = []

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope('parameters_Attention'):
            self.WS1 = tf.Variable(tf.truncated_normal([self.attention_depth, 2 * self.rnn_state_size], stddev=0.1),
                                   name="WS1")
            self.WS2 = tf.Variable(tf.truncated_normal([self.attention_hops, self.attention_depth], stddev=0.1),
                                   name="WS2")

            rnn_layers_fw = [tf.nn.rnn_cell.GRUCell(size) for size in ([self.rnn_state_size] * self.rnn_depth)]
            rnn_layers_bw = [tf.nn.rnn_cell.GRUCell(size) for size in ([self.rnn_state_size] * self.rnn_depth)]

            self.cell_fw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_fw)
            self.cell_bw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_bw)

        with tf.name_scope('Self-Attentive1'):
            self.function_1 = self.self_attentive_network(self.x_1, self.lengths_1)
        with tf.name_scope('Self-Attentive2'):
            self.function_2 = self.self_attentive_network(self.x_2, self.lengths_2)

        self.dense_1 = tf.nn.relu(tf.layers.dense(self.function_1, self.dense_layer_size))
        self.dense_2 = tf.nn.relu(tf.layers.dense(self.function_2, self.dense_layer_size))

        with tf.name_scope('Embedding1'):
            self.function_embedding_1 = tf.layers.dense(self.dense_1, self.embedding_size)
        with tf.name_scope('Embedding2'):
            self.function_embedding_2 = tf.layers.dense(self.dense_2, self.embedding_size)

        with tf.name_scope('siamese_layer'):
            self.cos_similarity = tf.reduce_sum(tf.multiply(self.function_embedding_1, self.function_embedding_2),
                                                axis=1,
                                                name="cosSimilarity")

            # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):
            A_square = tf.matmul(self.A, tf.transpose(self.A, perm=[0, 2, 1]))

            I = tf.eye(tf.shape(A_square)[1])
            I_tiled = tf.tile(tf.expand_dims(I, 0), [tf.shape(A_square)[0], 1, 1], name="I_tiled")
            self.A_pen = tf.norm(A_square - I_tiled)

            self.loss = tf.reduce_sum(tf.squared_difference(self.cos_similarity, self.y), name="loss")
            self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss + self.A_pen

            # Train step
        with tf.name_scope("Train_Step"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.regularized_loss)
