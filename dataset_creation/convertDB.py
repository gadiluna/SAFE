import sqlite3
import json
from networkx.readwrite import json_graph
import logging
from tqdm import tqdm
from asm_embedding.InstructionsConverter import InstructionsConverter


# Create the db where data are stored
def create_db(db_name):
    print('Database creation...')
    conn = sqlite3.connect(db_name)
    conn.execute(''' CREATE TABLE  IF NOT EXISTS functions (id INTEGER PRIMARY KEY, 
                                                            project text, 
                                                            compiler text, 
                                                            optimization text, 
                                                            file_name text, 
                                                            function_name text, 
                                                            asm text,
                                                            num_instructions INTEGER)
                ''')
    conn.execute('''CREATE TABLE  IF NOT EXISTS filtered_functions  (id INTEGER PRIMARY KEY, 
                                                                     instructions_list text)
                 ''')
    conn.commit()
    conn.close()


def reverse_graph(cfg, lstm_cfg):
    instructions = []
    asm = ""
    node_addr = list(cfg.nodes())
    node_addr.sort()
    nodes = cfg.nodes(data=True)
    lstm_nodes = lstm_cfg.nodes(data=True)
    for addr in node_addr:
        a = nodes[addr]["asm"]
        if a is not None:
            asm += a
        instructions.extend(lstm_nodes[addr]['features'])
    return instructions, asm


def copy_split(old_cur, new_cur, table):
    q = old_cur.execute("SELECT id FROM {}".format(table))
    iii = q.fetchall()
    print("Copying table {}".format(table))
    for ii in tqdm(iii):
        new_cur.execute("INSERT INTO {} VALUES (?)".format(table), ii)


def copy_table(old_cur, new_cur, table_old, table_new):
    q = old_cur.execute("SELECT * FROM {}".format(table_old))
    iii = q.fetchall()
    print("Copying table {} to {}".format(table_old, table_new))
    for ii in tqdm(iii):
        new_cur.execute("INSERT INTO {} VALUES (?,?,?)".format(table_new), ii)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

db = "/home/lucamassarelli/binary_similarity_data/databases/big_dataset_X86.db"
new_db = "/home/lucamassarelli/binary_similarity_data/new_databases/big_dataset_X86_new.db"

create_db(new_db)

conn_old = sqlite3.connect(db)
conn_new = sqlite3.connect(new_db)


cur_old = conn_old.cursor()
cur_new = conn_new.cursor()


q = cur_old.execute("SELECT id FROM functions")
ids = q.fetchall()
converter = InstructionsConverter()

for my_id in tqdm(ids):

    q0 = cur_old.execute("SELECT id, project, compiler, optimization, file_name, function_name, cfg FROM functions WHERE id=?", my_id)
    meta = q.fetchone()

    q1 = cur_old.execute("SELECT lstm_cfg FROM lstm_cfg WHERE id=?", my_id)
    cfg = json_graph.adjacency_graph(json.loads(meta[6]))
    lstm_cfg = json_graph.adjacency_graph(json.loads(q1.fetchone()[0]))
    instructions, asm = reverse_graph(cfg, lstm_cfg)
    values = meta[0:6] + (asm, len(instructions))
    q_n = cur_new.execute("INSERT INTO functions VALUES (?,?,?,?,?,?,?,?)", values)
    converted_instruction = json.dumps(converter.convert_to_ids(instructions))
    q_n = cur_new.execute("INSERT INTO filtered_functions VALUES (?,?)", (my_id[0], converted_instruction))

conn_new.commit()

cur_new.execute("CREATE TABLE train (id INTEGER PRIMARY KEY) ")
cur_new.execute("CREATE TABLE validation (id INTEGER PRIMARY KEY) ")
cur_new.execute("CREATE TABLE test (id INTEGER PRIMARY KEY) ")
conn_new.commit()

copy_split(cur_old, cur_new, "train")
conn_new.commit()
copy_split(cur_old, cur_new, "validation")
conn_new.commit()
copy_split(cur_old, cur_new, "test")
conn_new.commit()

cur_new.execute("CREATE TABLE  train_pairs (id INTEGER PRIMARY KEY, true_pair  TEXT, false_pair TEXT)")
cur_new.execute("CREATE TABLE  validation_pairs (id INTEGER PRIMARY KEY, true_pair  TEXT, false_pair TEXT)")
cur_new.execute("CREATE TABLE  test_pairs (id INTEGER PRIMARY KEY, true_pair  TEXT, false_pair TEXT)")
conn_new.commit()

copy_table(cur_old, cur_new, "train_couples", "train_pairs")
conn_new.commit()
copy_table(cur_old, cur_new, "validation_couples", "validation_pairs")
conn_new.commit()
copy_table(cur_old, cur_new, "test_couples", "test_pairs")
conn_new.commit()

conn_new.close()