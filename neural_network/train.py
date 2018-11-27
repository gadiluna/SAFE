from SAFE_model import modelSAFE
from parameters import Flags
import sys
import os
import numpy as np
from utils import utils
import traceback


def load_embedding_matrix(embedder_folder):
    matrix_file='embedding_matrix.npy'
    matrix_path=os.path.join(embedder_folder,matrix_file)
    if os.path.isfile(matrix_path):
        try:
            print('Loading embedding matrix....')
            with open(matrix_path,'rb') as f:
                return np.float32(np.load(f))
        except Exception as e:
            print("Exception handling file:"+str(matrix_path))
            print("Embedding matrix cannot be load")
            print(str(e))
            sys.exit(-1)

    else:
        print('Embedding matrix not found at path:'+str(matrix_path))
        sys.exit(-1)


def run_test():
    flags = Flags()
    flags.logger.info("\n{}\n".format(flags))

    print(str(flags))

    embedding_matrix = load_embedding_matrix(flags.embedder_folder)
    if flags.random_embedding:
        embedding_matrix = np.random.rand(*np.shape(embedding_matrix)).astype(np.float32)
        embedding_matrix[0, :] = np.zeros(np.shape(embedding_matrix)[1]).astype(np.float32)

    if flags.cross_val:
        print("STARTING CROSS VALIDATION")
        res = []
        mean = 0
        for i in range(0, flags.cross_val_fold):
            print("CROSS VALIDATION STARTING FOLD: " + str(i))
            if i > 0:
                flags.close_log()
                flags.reset_logdir()
                del flags
                flags = Flags()
                flags.logger.info("\n{}\n".format(flags))

            flags.logger.info("Starting cross validation fold: {}".format(i))

            flags.db_name = flags.db_name + "_val_" + str(i+1) + ".db"
            flags.logger.info("Cross validation db name: {}".format(flags.db_name))

            trainer = modelSAFE(flags, embedding_matrix)
            best_val_auc = trainer.train()

            mean += best_val_auc
            res.append(best_val_auc)

            flags.logger.info("Cross validation fold {} finished best auc: {}".format(i, best_val_auc))
            print("FINISH FOLD: " + str(i) + " BEST VAL AUC: " + str(best_val_auc))

        print("CROSS VALIDATION ENDED")
        print("Result: " + str(res))
        print("")

        flags.logger.info("Cross validation finished results: {}".format(res))
        flags.logger.info(" mean: {}".format(mean / flags.cross_val_fold))
        flags.close_log()

    else:
        trainer = modelSAFE(flags, embedding_matrix)
        trainer.train()
        flags.close_log()


if __name__ == '__main__':
    utils.print_safe()
    print('-Trainer for SAFE-')
    run_test()
