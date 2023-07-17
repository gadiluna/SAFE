# SAFE TEAM
# Copyright (C) 2019  Luca Massarelli, Giuseppe Antonio Di Luna, Fabio Petroni, Leonardo Querzoni, Roberto Baldoni

import numpy as np


class FunctionNormalizer:

    def __init__(self, max_instruction):
        self.max_instructions = max_instruction

    def normalize(self, f):
        f = np.asarray(f[0:self.max_instructions])
        length = f.shape[0]
        if f.shape[0] < self.max_instructions:
            f = np.pad(f, (0, self.max_instructions - f.shape[0]), mode='constant')
        return f, length

    def normalize_function_pairs(self, pairs):
        lengths = []
        new_pairs = []
        for x in pairs:
            f0, len0 = self.normalize(x[0])
            f1, len1 = self.normalize(x[1])
            lengths.append((len0, len1))
            new_pairs.append((f0, f1))
        return new_pairs, lengths

    def normalize_functions(self, functions):
        lengths = []
        new_functions = []
        for f in functions:
            f, length = self.normalize(f)
            lengths.append(length)
            new_functions.append(f)
        return new_functions, lengths
