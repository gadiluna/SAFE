# SAFE : Self Attentive Function Embedding
  
What you need  
-----
You need radare2 installed in your system. 
  
Quickstart
-----
To create the embedding of a function:
```
git clone repo url
pip install -r requirements
chmod +x download_model.sh
./download_model.sh
python safe.py -m data/safe.pb -i helloworld.o -a 100000F30
```
#### What to do with an embedding?
Once you have two embeddings ```embedding_x``` and ```embedding_y``` you can compute the similarity of the corresponding functions as: 
```
from sklearn.metrics.pairwise import cosine_similarity

sim=cosine_similarity(embedding_x, embedding_y)
 
```


Data Needed
-----
SAFE needs few information to work. Two are essentials, a model that tells safe how to 
convert assembly instructions in vectors (i2v model) and a model that tells safe how
to convert an binary function into a vector.
Both models can be downloaded by using the command
```
./download_model.sh
```
the downloader downloads the model and place them in the directory data.
The directory tree after the download should be.
```
safe/-- githubcode
     \
      \--data/-----safe.pb
               \
                \---i2v/
            
```
The safe.pb file contains the safe-model used to convert binary function to vectors.
The i2v folder contains the i2v model. 


Hardcore Details
----
This section contains details that are needed to replicate our experiments, if you are an user of safe you can skip
it. 

### Safe.pb
This is the freezed tensorflow trained model for AMD64 architecture. You can import it in your project using:

```
 import tensorflow as tf
 
 with tf.gfile.GFile("safe.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

 with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
    
 sess = tf.Session(graph=graph)
``` 

see file: neural_network/SAFEEmbedder.py

### i2v
The i2v folder contains two files. 
A Matrix where each row is the embedding of an asm instruction.
A json file that contains a dictonary mapping asm instructions into row numbers of the matrix above.
see file: asm_embedding/InstructionsConverter.py



## Train the model
If you want to train the model using our datasets you have to first use:
```
 python3 downloader.py -tr
```
This will download the datasets into data folder. Note that the datasets are compressed so you have to decompress them yourself.
This data will be an sqlite databases.
To start the train use neural_network/train.sh.
The db can be selected by changing the parameter into train.sh.
If you want information on the dataset see our paper.

## Create your own dataset
If you want to create your own dataset you can use the script ExperimentUtil into the folder
dataset creation.

## Create a functions knowledge base
If you want to use SAFE binary code search engine you can use the script ExperimentUtil to create
the knowledge base.
Then you can search through it using the script into function_search

Paper
---
See our arXiv paper: [arxiv](https://arxiv.org/abs/1811.05296)