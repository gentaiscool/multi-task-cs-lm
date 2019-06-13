## Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

The implementation of <a href="http://aclweb.org/anthology/W18-3207">Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning</a> (3rd Workshop in Computational Approaches in Linguistic Code-switching, ACL 2018) paper. The code is written in Python using Pytorch.

Supplementary Materials (including the distribution of train, dev, and test) can be found <a href="https://github.com/gentaiscool/multi-task-cs-lm/blob/master/doc/supplementary-materials-code.pdf">here</a>.

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@InProceedings{W18-3207,
  author = 	"Winata, Genta Indra
		and Madotto, Andrea
		and Wu, Chien-Sheng
		and Fung, Pascale",
  title = 	"Code-Switching Language Modeling using Syntax-Aware Multi-Task Learning",
  booktitle = 	"Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"62--67",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/W18-3207"
}
</pre>

## Abstract
Lack of text data has been the major issue on code-switching language modeling. In this paper, we introduce multi-task learning based language model which shares syntax representation of languages to leverage linguistic information and tackle the low resource data issue. Our model jointly learns both language modeling and Part-of-Speech tagging on code-switched utterances. In this way, the model is able to identify the location of code-switching points and improves the prediction of next word. Our approach outperforms standard LSTM based language model, with an improvement of 9.7% and 7.4% in perplexity on SEAME Phase I and Phase II dataset respectively.

## Model Architecture
<img src="img/multi-task-model.jpg" width=300>

## Prerequisites:
- Python 3.5 or 3.6
- Pytorch 0.2 (or later)
- Stanford Core NLP (Tokenization and Segmentation)

## Data
SEAME Corpus from LDC: <a href="https://catalog.ldc.upenn.edu/ldc2015s04">Mandarin-English Code-Switching in South-East Asia</a>

## Run the code:

<b>Multi-task</b>
```console
❱❱❱ python main_multi_task.py --tied --clip=0.25 --dropout=0.4 --postagdropout=0.4 --p=0.25 --nhid=500 --postagnhid=500 --emsize=500 --postagemsize=500 --cuda --data=../data/seame_phase2
```
