# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import json
import random
import sqlite3
from tqdm import tqdm


class DataSplitter:

    def __init__(self, db_name):
        self.db_name = db_name

    def create_pair_table(self, table_name):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.executescript("DROP TABLE IF EXISTS {} ".format(table_name))
        c.execute("CREATE TABLE  {} (id INTEGER PRIMARY KEY, true_pair  TEXT, false_pair TEXT)".format(table_name))
        conn.commit()
        conn.close()

    def get_ids(self, set_type):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        q = cur.execute("SELECT id FROM {}".format(set_type))
        ids = q.fetchall()
        conn.close()
        return ids

    @staticmethod
    def select_similar_cfg(id, provenance, ids, cursor):
        q1 = cursor.execute('SELECT id FROM functions WHERE project=? AND file_name=? and function_name=?', provenance)
        candidates = [i[0] for i in q1.fetchall() if (i[0] != id and i[0] in ids)]
        if len(candidates) == 0:
            return None
        id_similar = random.choice(candidates)
        return id_similar

    @staticmethod
    def select_dissimilar_cfg(ids, provenance, cursor):
        while True:
            id_dissimilar = random.choice(ids)
            q2 = cursor.execute('SELECT project, file_name, function_name FROM functions WHERE id=?', id_dissimilar)
            res = q2.fetchone()
            if res != provenance:
                break
        return id_dissimilar

    def create_epoch_pairs(self, epoch_number, pairs_table,id_table):
        random.seed = epoch_number

        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        ids = cur.execute("SELECT id FROM "+id_table).fetchall()
        id_set=set(ids)
        true_pair = []
        false_pair = []

        for my_id in tqdm(ids):
            q = cur.execute('SELECT project, file_name, function_name FROM functions WHERE id =?', my_id)
            cfg_0_provenance = q.fetchone()
            id_sim = DataSplitter.select_similar_cfg(my_id, cfg_0_provenance, id_set, cur)
            id_dissim = DataSplitter.select_dissimilar_cfg(ids, cfg_0_provenance, cur)
            if id_sim is not None and id_dissim is not None:
                true_pair.append((my_id, id_sim))
                false_pair.append((my_id, id_dissim))

        true_pair = str(json.dumps(true_pair))
        false_pair = str(json.dumps(false_pair))

        cur.execute("INSERT INTO {} VALUES (?,?,?)".format(pairs_table), (epoch_number, true_pair, false_pair))
        conn.commit()
        conn.close()

    def create_pairs(self, total_epochs):

        self.create_pair_table('train_pairs')
        self.create_pair_table('validation_pairs')
        self.create_pair_table('test_pairs')

        for i in range(0, total_epochs):
            print("Creating training pairs for epoch {} of {}".format(i, total_epochs))
            self.create_epoch_pairs(i, 'train_pairs','train')

        print("Creating validation pairs")
        self.create_epoch_pairs(0, 'validation_pairs','validation')

        print("Creating test pairs")
        self.create_epoch_pairs(0, "test_pairs",'test')


    @staticmethod
    def prepare_set(data_to_include, table_name, file_list, cur):
        i = 0
        while i < data_to_include and len(file_list) > 0:
            choice = random.choice(file_list)
            file_list.remove(choice)
            q = cur.execute("SELECT id FROM functions where project=? AND file_name=?", choice)
            data = q.fetchall()
            cur.executemany("INSERT INTO {} VALUES (?)".format(table_name), data)
            i += len(data)
        return file_list, i

    def split_data(self, validation_dim, test_dim):
        random.seed = 12345
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        q = c.execute('''SELECT project, file_name FROM functions ''')
        data = q.fetchall()
        conn.commit()

        num_data = len(data)
        num_test = int(num_data * test_dim)
        num_validation = int(num_data * validation_dim)

        filename = list(set(data))

        c.execute("DROP TABLE IF EXISTS train")
        c.execute("DROP TABLE IF EXISTS test")
        c.execute("DROP TABLE IF EXISTS validation")

        c.execute("CREATE TABLE IF NOT EXISTS train (id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE IF NOT EXISTS validation (id INTEGER PRIMARY KEY)")
        c.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")

        c.execute('''CREATE INDEX IF NOT EXISTS  my_index   ON functions(project, file_name, function_name)''')
        c.execute('''CREATE INDEX IF NOT EXISTS  my_index_2 ON functions(project, file_name)''')

        filename, test_num = DataSplitter.prepare_set(num_test, 'test', filename, conn.cursor())
        conn.commit()
        assert len(filename) > 0
        filename, val_num = self.prepare_set(num_validation, 'validation', filename, conn.cursor())
        conn.commit()
        assert len(filename) > 0
        _, train_num = self.prepare_set(num_data - num_test - num_validation, 'train', filename, conn.cursor())
        conn.commit()

        print("Train Size: {}".format(train_num))
        print("Validation Size:  {}".format(val_num))
        print("Test Size: {}".format(test_num))
