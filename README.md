# NLP_Exam_project-2023

***Practical product for Natural Language Processing Exam project***
***at Aarhus University***

## About the project
This repository presents a Many-to-many LSTM neural network with a word embedding layer to predict the subsequent dance move in Lindy Hop. This product was built as part of the Natural language processing exam at Aarhus University.  It aims to contribute to a better and more evidence-based understanding of the dance language. This product primarily serves as a research tool to investigate how dance move labels represented as Word2Vec word embeddings could inform about the inherent dance structure and, thus, help investigate the syntax of dance. It could be used to study different cultural influences on dance, and investigate how the language of dance changed over time. Once dance move prediction is successful, possible applications could include computer-aided dance teaching and learning. Also it could be utilized in arts and entertainment for a generation of human-like creative movement. 

## About the script
This script builds and trains a Many-to-many LSTM neural network with a word embedding layer for the Lindy Hop dance sequence prediction task. The code could be re-used with any other dance dataset with the same structure. The dance move sequences should be saved as .csv file with a 'Behavior'column where each row is a separate dance sequence of labeled moves separated by a comma. The word embedding matrix was created for this project´s dataset using Word2Vec.

## Acknowledgement
It is essential to acknowledge that Lindy Hop is a vibrant and expressive art form. This research does not attempt to diminish its richness by translating it into mere numbers. Instead, by employing contemporary computational methods, this study aims to enhance the evolution of Lindy Hop, offering new insights for its research, historical appreciation, and educational endeavors.


## Repository structure

| File | Description |
| --- | --- |
| data/ | Folder containing files input data for the script |
| out/ | Folder containing files produced by the scripts |
| src/ | Folder containing the scripts |
| LICENSE | A software license defining what other users can and can't do with the source code |
| README.md | Description of the project and the instructions |
| create_NP_venv.bash | bash file for creating a virtual environment |
| kill_NLP_venv.bash | bash file for removing a virtual environment |
| requirements.txt | list of Python packages required to run the script |

## Instructions to run the code

The code was tested on an HP computer with Windows 11 operating system with Python 3.11 (64-bit) 
Note: It required Installing CUDA and its version. Installing cuDNN, cuFFT, cuBLAS, and their versions.

__Steps__

Set-up:
```
#1 Open terminal
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/NLP_Exam_project-2023.git

#4 Navigate to the newly cloned repo
$ cd NLP_Exam_project-2023

#5 Create a virtual environment with its dependencies and activate it
$ bash create_NLP_venv.sh
$ source ./NLP/bin/activate

```

Run the code:

```
#6 Navigate to the directory of the scripts
$ cd src

#7 Run the code with default parameters
$ python Emb_LSTM.py -dir ../data/LindyHop_moves_sequences.csv -emb_m ../data/embedding_matrix_word2vec_100.npy

#8 To remove the newly created virtual environment
$ bash kill_NLP_venv.sh

#9 To find out possible optional arguments for both scripts
$ python Emb_LSTM.py --help



 ```
