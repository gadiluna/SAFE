import tensorflow as tf
# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt)
#

class SAFEEmbedder:

    def __init__(self, model_file):
        self.model_file = model_file
        self.session = None
        self.x_1 = None
        self.adj_1 = None
        self.len_1 = None
        self.emb = None

    def loadmodel(self):
        with tf.gfile.GFile(self.model_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        sess = tf.Session(graph=graph)
        self.session = sess

        return sess

    def get_tensor(self):
        self.x_1 = self.session.graph.get_tensor_by_name("import/x_1:0")
        self.len_1 = self.session.graph.get_tensor_by_name("import/lengths_1:0")
        self.emb = tf.nn.l2_normalize(self.session.graph.get_tensor_by_name('import/Embedding1/dense/BiasAdd:0'), axis=1)

    def embedd(self, nodi_input, lengths_input):

        out_embedding= self.session.run(self.emb, feed_dict = {
                                                    self.x_1: nodi_input,
                                                    self.len_1: lengths_input})

        return out_embedding
