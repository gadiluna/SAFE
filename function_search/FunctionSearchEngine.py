# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/
import sys
import numpy as np
import sqlite3
import pandas as pd
import tqdm
import tensorflow as tf

if sys.version_info >= (3, 0):
    from functools import reduce


pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

class TopK:

    #
    # This class computes the similarities between the targets and the list of functions on which we are searching.
    # This is done by using matrices multiplication and top_k of tensorflow
    def __init__(self):
        self.graph=tf.Graph()
        nop=0

    def loads_embeddings_SE(self, lista_embeddings):
        with self.graph.as_default():
            tf.set_random_seed(1234)
            dim = lista_embeddings[0].shape[0]
            ll = np.asarray(lista_embeddings)
            self.matrix = tf.constant(ll, name='matrix_embeddings', dtype=tf.float32)
            self.target = tf.placeholder("float", [None, dim], name='target_embedding')
            self.sim = tf.matmul(self.target, self.matrix, transpose_b=True, name="embeddings_similarities")
            self.k = tf.placeholder(tf.int32, shape=(), name='k')
            self.top_k = tf.nn.top_k(self.sim, self.k, sorted=True)
            self.session = tf.Session()

    def topK(self, k, target):
        with self.graph.as_default():
            tf.set_random_seed(1234)
            return self.session.run(self.top_k, {self.target: target, self.k: int(k)})

class FunctionSearchEngine:

    def __init__(self, db_name, table_name, limit=None):
        self.s2v = TopK()
        self.db_name = db_name
        self.table_name = table_name
        self.labels = []
        self.trunc_labels = []
        self.lista_embedding = []
        self.ids = []
        self.n_similar=[]
        self.ret = {}
        self.precision = None

        print("Query for ids")
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        if limit is None:
            q = cur.execute("SELECT id, project, compiler, optimization, file_name, function_name FROM functions")
            res = q.fetchall()
        else:
            q = cur.execute("SELECT id, project, compiler, optimization, file_name, function_name FROM functions LIMIT {}".format(limit))
            res = q.fetchall()

        for item in tqdm.tqdm(res, total=len(res)):
            q = cur.execute("SELECT " + self.table_name + " FROM " + self.table_name + " WHERE id=?", (item[0],))
            e = q.fetchone()
            if e is None:
                continue

            self.lista_embedding.append(self.embeddingToNp(e[0]))

            element = "{}/{}/{}".format(item[1], item[4], item[5])
            self.trunc_labels.append(element)

            element = "{}@{}/{}/{}/{}".format(item[5], item[1], item[2], item[3], item[4])
            self.labels.append(element)
            self.ids.append(item[0])

        conn.close()

        self.s2v.loads_embeddings_SE(self.lista_embedding)
        self.num_funcs = len(self.lista_embedding)

    def load_target(self, target_db_name, target_fcn_ids, calc_mean=False):
        conn = sqlite3.connect(target_db_name)
        cur = conn.cursor()
        mean = None
        for id in target_fcn_ids:

            if target_db_name == self.db_name and id in self.ids:
                idx = self.ids.index(id)
                e = self.lista_embedding[idx]
            else:
                q = cur.execute("SELECT " + self.table_name + " FROM " + self.table_name + " WHERE id=?", (id,))
                e = q.fetchone()
                e = self.embeddingToNp(e[0])


            if mean is None:
                mean = e.reshape([e.shape[0], 1])
            else:
                mean = np.hstack((mean, e.reshape(e.shape[0], 1)))

        if calc_mean:
            target = [np.mean(mean, axis=1)]
        else:
            target = mean.T
        return target

    def embeddingToNp(self, e):
        e = e.replace('\n', '')
        e = e.replace('[', '')
        e = e.replace(']', '')
        emb = np.fromstring(e, dtype=float, sep=' ')
        return emb

    def top_k(self, target, k=None):
        if k is not None:
            top_k = self.s2v.topK(k, target)
        else:
            top_k = self.s2v.topK(len(self.lista_embedding), target)
        return top_k

    def pp_search(self, k):
        result = pd.DataFrame(columns=['Id', 'Name', 'Score'])
        top_k = self.s2v.topK(k)
        for i, e in enumerate(top_k.indices[0]):
            result = result.append({'Id': self.ids[e], 'Name': self.labels[e], 'Score': top_k.values[0][i]}, ignore_index=True)
        print(result)

    def search(self, k):
        result = []
        top_k = self.s2v.topK(k)
        for i, e in enumerate(top_k.indices[0]):
            result = result.append({'Id': self.ids[e], 'Name': self.labels[e], 'Score': top_k.values[0][i]})
        return result
