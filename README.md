# Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning
The implementation of <a href="https://arxiv.org/abs/1805.12070">Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning</a> (3rd Workshop in Computational Approaches in Linguistic Code-switching, ACL 2018) paper. The code is written in Python using Pytorch.

<img src="img/multi-task-model.jpg" width=300>

## Prerequisites:
- Python 3.X
- Pytorch 0.2.X (or later)
- Stanford Core NLP (Tokenization and Segmentation)

## Data
SEAME Corpus from LDC: <a href="https://catalog.ldc.upenn.edu/ldc2015s04">Mandarin-English Code-Switching in South-East Asia</a>

## Run the code:

<b>Multi-task</b>
```
python main_multi_task.py --tied --clip=0.25 --dropout=0.4 --postagdropout=0.4 --p=0.25 --nhid=500 --postagnhid=500 --emsize=500 --postagemsize=500 --cuda --data=../data/seame_phase2
```
