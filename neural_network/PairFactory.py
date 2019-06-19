# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/
import sqlite3

import json
import numpy as np

from multiprocessing import Queue
from multiprocessing import Process
from asm_embedding.FunctionNormalizer import FunctionNormalizer

#
# PairFactory class, used for training the SAFE network.
# This class generates the pairs for training, test and validation
#
#
# Authors: SAFE team


class PairFactory:

    def __init__(self, db_name, dataset_type, batch_size, max_instructions, shuffle=True):
        self.db_name = db_name
        self.dataset_type = dataset_type
        self.max_instructions = max_instructions
        self.batch_dim = 0
        self.num_pairs = 0
        self.num_batches = 0
        self.batch_size = batch_size
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q = cur.execute("SELECT true_pair from " + self.dataset_type + " WHERE id=?", (0,))
        self.num_pairs=len(json.loads(q.fetchone()[0]))*2
        n_chunk = int(self.num_pairs / self.batch_size) - 1
        conn.close()
        self.num_batches = n_chunk
        self.shuffle = shuffle

    @staticmethod
    def split( a, n):
        return [a[i::n] for i in range(n)]

    @staticmethod
    def truncate_and_compute_lengths(pairs, max_instructions):
        lenghts = []
        new_pairs=[]
        for x in pairs:
            f0 = np.asarray(x[0][0:max_instructions])
            f1 = np.asarray(x[1][0:max_instructions])
            lenghts.append((f0.shape[0], f1.shape[0]))
            if f0.shape[0] < max_instructions:
                f0 = np.pad(f0, (0, max_instructions - f0.shape[0]), mode='constant')
            if f1.shape[0] < max_instructions:
                f1 = np.pad(f1, (0, max_instructions - f1.shape[0]), mode='constant')

            new_pairs.append((f0, f1))
        return new_pairs, lenghts

    def async_chunker(self, epoch):

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        query_string = "SELECT true_pair,false_pair from {} where id=?".format(self.dataset_type)
        q = cur.execute(query_string, (int(epoch),))
        true_pairs_id, false_pairs_id = q.fetchone()
        true_pairs_id = json.loads(true_pairs_id)
        false_pairs_id = json.loads(false_pairs_id)

        assert len(true_pairs_id) == len(false_pairs_id)
        data_len = len(true_pairs_id)

        # print("Data Len: " + str(data_len))
        conn.close()

        n_chunk = int(data_len / (self.batch_size / 2)) - 1
        lista_chunk = range(0, n_chunk)
        coda = Queue(maxsize=50)
        n_proc = 8  # modify this to increase the parallelism for the db loading, from our thest 8-10 is the sweet spot on a 16 cores machine with K80
        listone = PairFactory.split(lista_chunk, n_proc)

        # this ugly workaround is somehow needed, Pool is working oddly when TF is loaded.
        for i in range(0, n_proc):
            p = Process(target=self.async_create_couple, args=((epoch, listone[i], coda)))
            p.start()

        for i in range(0, n_chunk):
            yield self.async_get_dataset(coda)

    def get_pair_fromdb(self, id_1, id_2):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q0 = cur.execute("SELECT instructions_list FROM filtered_functions WHERE id=?", (id_1,))
        f0 = json.loads(q0.fetchone()[0])

        q1 = cur.execute("SELECT instructions_list FROM filtered_functions WHERE id=?", (id_2,))
        f1 = json.loads(q1.fetchone()[0])
        conn.close()
        return f0, f1

    def get_couple_from_db(self, epoch_number, chunk):

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()

        pairs = []
        labels = []

        q = cur.execute("SELECT true_pair, false_pair from " + self.dataset_type + " WHERE id=?", (int(epoch_number),))
        true_pairs_id, false_pairs_id = q.fetchone()

        true_pairs_id = json.loads(true_pairs_id)
        false_pairs_id = json.loads(false_pairs_id)
        conn.close()
        data_len = len(true_pairs_id)

        i = 0

        normalizer = FunctionNormalizer(self.max_instructions)

        while i < self.batch_size:
            if chunk * int(self.batch_size / 2) + i > data_len:
                break

            p = true_pairs_id[chunk * int(self.batch_size / 2) + i]
            f0, f1 = self.get_pair_fromdb(p[0], p[1])
            pairs.append((f0, f1))
            labels.append(+1)

            p = false_pairs_id[chunk * int(self.batch_size / 2) + i]
            f0, f1 = self.get_pair_fromdb(p[0], p[1])
            pairs.append((f0, f1))
            labels.append(-1)

            i += 2

        pairs, lengths = normalizer.normalize_function_pairs(pairs)

        function1, function2 = zip(*pairs)
        len1, len2 = zip(*lengths)
        n_samples = len(pairs)

        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(n_samples))

            function1 = np.array(function1)[shuffle_indices]

            function2 = np.array(function2)[shuffle_indices]
            len1 = np.array(len1)[shuffle_indices]
            len2 = np.array(len2)[shuffle_indices]
            labels = np.array(labels)[shuffle_indices]
        else:
            function1=np.array(function1)
            function2=np.array(function2)
            len1=np.array(len1)
            len2=np.array(len2)
            labels=np.array(labels)

        upper_bound = min(self.batch_size, n_samples)
        len1 = len1[0:upper_bound]
        len2 = len2[0:upper_bound]
        function1 = function1[0:upper_bound]
        function2 = function2[0:upper_bound]
        y_ = labels[0:upper_bound]
        return function1, function2, len1, len2, y_

    def async_create_couple(self, epoch,n_chunk,q):
        for i in n_chunk:
            function1, function2, len1, len2, y_ = self.get_couple_from_db(epoch, i)
            q.put((function1, function2, len1, len2, y_), block=True)

    def async_get_dataset(self, q):

        item = q.get()
        function1 = item[0]
        function2 = item[1]
        len1 = item[2]
        len2 = item[3]
        y_ = item[4]

        assert (len(function1) == len(y_))
        n_samples = len(function1)
        self.batch_dim = n_samples
        #self.num_pairs += n_samples

        return function1, function2, len1, len2, y_

