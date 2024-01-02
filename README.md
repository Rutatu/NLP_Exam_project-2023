# NLP_Exam_project-2023

***Exam project practical product for Natural Language Processing MA´s course at Aarhus University.***


## Intructions to run the code

The code was tested on an HP computer with Windows 10 operating system. 

__Steps__

Set-up:
```
#1 Open terminal
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/cds-language_Assignment_6.git 

#4 Navigate to the newly cloned repo
$ cd cds-language_Assignment_6

#5 Create virtual environment with its dependencies and activate it
$ bash create_NLP_venv.sh
$ source ./NLP/bin/activate

```

Run the code:

```
#6 Navigate to the directory of the scripts
$ cd src

#7 Run each code with default parameters
$ python GoT_LogReg.py -dir ../data/Game_of_Thrones_Script.csv
$ python GoT_deep.py -dir ../data/Game_of_Thrones_Script.csv 

#8 To remove the newly created virtual environment
$ bash kill_NLP_venv.sh

#9 To find out possible optional arguments for both scripts
$ python GoT_LogReg.py --help
$ python GoT_deep.py --help


 ```
