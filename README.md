The implementation of <a href="https://arxiv.org/abs/1805.12070">Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning</a> ("3rd Workshop in Computational Approaches in Linguistic Code-switching", ACL 2018) paper. The code is written in Python using Pytorch.

### Prerequisites:
- Python 3.X
- Pytorch 0.2.X (or later)

### Run the code:

<b>Multi-task</b>
```
python3 main_multi_task.py --tied --clip=0.25 --dropout=0.4 --postagdropout=0.4 --alpha=0.25 --beta=0.75 --nhid=500 --postagnhid=500 --emsize=500 --postagemsize=500 --cuda --data=../data/seame_phase2
```
