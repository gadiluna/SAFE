# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
from asm_embedding.FunctionNormalizer import FunctionNormalizer
import json
from neural_network.SAFEEmbedder import SAFEEmbedder
import numpy as np
import sqlite3
from tqdm import tqdm


class FunctionsEmbedder:

    def __init__(self,  model, batch_size, max_instruction):
        self.batch_size = batch_size
        self.normalizer = FunctionNormalizer(max_instruction)
        self.safe = SAFEEmbedder(model)
        self.safe.loadmodel()
        self.safe.get_tensor()

    def compute_embeddings(self, functions):
        functions, lenghts = self.normalizer.normalize_functions(functions)
        embeddings = self.safe.embedd(functions, lenghts)
        return embeddings

    @staticmethod
    def create_table(db_name, table_name):
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, {}  TEXT)".format(table_name, table_name))
        conn.commit()
        conn.close()

    def compute_and_save_embeddings_from_db(self, db_name, table_name):
        FunctionsEmbedder.create_table(db_name, table_name)
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        q = cur.execute("SELECT id FROM functions WHERE id not in (SELECT id from {})".format(table_name))
        ids = q.fetchall()

        for i in tqdm(range(0, len(ids), self.batch_size)):
            functions = []
            batch_ids = ids[i:i+self.batch_size]
            for my_id in batch_ids:
                q = cur.execute("SELECT instructions_list FROM filtered_functions where id=?", my_id)
                functions.append(json.loads(q.fetchone()[0]))
            embeddings = self.compute_embeddings(functions)

            for l, id in enumerate(batch_ids):
                cur.execute("INSERT INTO {} VALUES (?,?)".format(table_name), (id[0], np.array2string(embeddings[l])))
            conn.commit()
