# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import json


class InstructionsConverter:

    def __init__(self, json_i2id):
        f = open(json_i2id, 'r')
        self.i2id = json.load(f)
        f.close()

    def convert_to_ids(self, instructions_list):
        ret_array = []
        # For each instruction we add +1 to its ID because the first
        # element of the embedding matrix is zero
        for x in instructions_list:
            if x in self.i2id:
                ret_array.append(self.i2id[x] + 1)
            elif 'X_' in x:
                # print(str(x) + " is not a known x86 instruction")
                ret_array.append(self.i2id['X_UNK'] + 1)
            elif 'A_' in x:
                # print(str(x) + " is not a known arm instruction")
                ret_array.append(self.i2id['A_UNK'] + 1)
            else:
                # print("There is a problem " + str(x) + " does not appear to be an asm or arm instruction")
                ret_array.append(self.i2id['X_UNK'] + 1)
        return ret_array


