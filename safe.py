# SAFE TEAM
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from asm_embedding.FunctionAnalyzerRadare import RadareFunctionAnalyzer
from argparse import ArgumentParser
from asm_embedding.FunctionNormalizer import FunctionNormalizer
from asm_embedding.InstructionsConverter import InstructionsConverter
from neural_network.SAFEEmbedder import SAFEEmbedder
from utils import utils

class SAFE:

    def __init__(self, model):
        self.converter = InstructionsConverter("data/i2v/word2id.json")
        self.normalizer = FunctionNormalizer(max_instruction=150)
        self.embedder = SAFEEmbedder(model)
        self.embedder.loadmodel()
        self.embedder.get_tensor()

    def embedd_function(self, filename, address):
        analyzer = RadareFunctionAnalyzer(filename, use_symbol=False, depth=0)
        functions = analyzer.analyze()
        instructions_list = None
        for function in functions:
            if functions[function]['address'] == address:
                instructions_list = functions[function]['filtered_instructions']
                break
        if instructions_list is None:
            print("Function not found")
            return None
        converted_instructions = self.converter.convert_to_ids(instructions_list)
        instructions, length = self.normalizer.normalize_functions([converted_instructions])
        embedding = self.embedder.embedd(instructions, length)
        return embedding


if __name__ == '__main__':

    utils.print_safe()

    parser = ArgumentParser(description="Safe Embedder")

    parser.add_argument("-m", "--model",   help="Safe trained model to generate function embeddings")
    parser.add_argument("-i", "--input",   help="Input executable that contains the function to embedd")
    parser.add_argument("-a", "--address", help="Hexadecimal address of the function to embedd")

    args = parser.parse_args()

    address = int(args.address, 16)
    safe = SAFE(args.model)
    embedding = safe.embedd_function(args.input, address)
    print(embedding[0])



