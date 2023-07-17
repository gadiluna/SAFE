# SAFE TEAM
# Copyright (C) 2019  Luca Massarelli, Giuseppe Antonio Di Luna, Fabio Petroni, Leonardo Querzoni, Roberto Baldoni


import argparse
import os
import sys
from subprocess import call

class Downloader:

    def __init__(self):
        parser = argparse.ArgumentParser(description='SAFE downloader')

        parser.add_argument("-m", "--model", dest="model", help="Download the trained SAFE model for x86",
                            action="store_true",
                            required=False)

        parser.add_argument("-i2v", "--i2v", dest="i2v", help="Download the i2v dictionary and embedding matrix",
                            action="store_true",
                            required=False)

        parser.add_argument("-b", "--bundle", dest="bundle",
                            help="Download all files necessary to run the model",
                            action="store_true",
                            required=False)

        parser.add_argument("-td", "--train_data", dest="train_data",
                            help="Download the files necessary to train the model (It takes a lot of space!)",
                            action="store_true",
                            required=False)

        args = parser.parse_args()

        self.download_model = (args.model or args.bundle)
        self.download_i2v = (args.i2v or args.bundle)
        self.download_train = args.train_data

        if not (self.download_model or self.download_i2v or self.download_train):
            parser.print_help(sys.__stdout__)

        self.url_model = "https://drive.google.com/file/d/1Kwl8Jy-g9DXe1AUjUZDhJpjRlDkB4NBs/view?usp=sharing"
        self.url_i2v = "https://drive.google.com/file/d/1CqJVGYbLDEuJmJV6KH4Dzzhy-G12GjGP"
        self.url_train = ['https://drive.google.com/file/d/1sNahtLTfZY5cxPaYDUjqkPTK0naZ45SH/view?usp=sharing','https://drive.google.com/file/d/16D5AVDux_Q8pCVIyvaMuiL2cw2V6gtLc/view?usp=sharing','https://drive.google.com/file/d/1cBRda8fYdqHtzLwstViuwK6U5IVHad1N/view?usp=sharing']
        self.train_name = ['AMD64ARMOpenSSL.tar.bz2','AMD64multipleCompilers.tar.bz2','AMD64PostgreSQL.tar.bz2']
        self.base_path = "data"
        self.path_i2v = os.path.join(self.base_path, "")
        self.path_model = os.path.join(self.base_path, "")
        self.path_train_data = os.path.join(self.base_path, "")
        self.i2v_compress_name='i2v.tar.bz2'
        self.model_compress_name='model.tar.bz2'
        self.datasets_compress_name='safe.pb'

    @staticmethod
    def download_file(id,path):
        try:
            print("Downloading from "+ str(id) +" into "+str(path))
            call(['./godown.pl',id,path])
        except Exception as e:
            print("Error downloading file at url:" + str(id))
            print(e)

    @staticmethod
    def decompress_file(file_src,file_path):
        try:
            call(['tar','-xvf',file_src,'-C',file_path])
        except Exception as e:
            print("Error decompressing file:" + str(file_src))
            print('you need tar command e b2zip support')
            print(e)

    def download(self):
        print('Making the godown.pl script executable, thanks:'+str('https://github.com/circulosmeos/gdown.pl'))
        call(['chmod', '+x','godown.pl'])
        print("SAFE --- downloading models")

        if self.download_i2v:
            print("Downloading i2v model.... in the folder data/i2v/")
            if not os.path.exists(self.path_i2v):
                os.makedirs(self.path_i2v)
            Downloader.download_file(self.url_i2v, os.path.join(self.path_i2v,self.i2v_compress_name))
            print("Decompressing i2v model and placing in" + str(self.path_i2v))
            Downloader.decompress_file(os.path.join(self.path_i2v,self.i2v_compress_name),self.path_i2v)

        if self.download_model:
            print("Downloading the SAFE model... in the folder data")
            if not os.path.exists(self.path_model):
                os.makedirs(self.path_model)
            Downloader.download_file(self.url_model, os.path.join(self.path_model,self.datasets_compress_name))
            #print("Decompressing SAFE model and placing in" + str(self.path_model))
            #Downloader.decompress_file(os.path.join(self.path_model,self.model_compress_name),self.path_model)

        if self.download_train:
            print("Downloading the train data.... in the folder data")
            if not os.path.exists(self.path_train_data):
                os.makedirs(self.path_train_data)
            for i,x in enumerate(self.url_train):
                print("Downloading dataset "+str(self.train_name[i]))
                Downloader.download_file(x, os.path.join(self.path_train_data,self.train_name[i]))
            #print("Decompressing the train data and placing in" + str(self.path_train_data))
            #Downloader.decompress_file(os.path.join(self.path_train_data,self.datasets_compress_name),self.path_train_data)

if __name__=='__main__':
    a=Downloader()
    a.download()