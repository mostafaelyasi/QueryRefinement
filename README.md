# QueryRefinement
Query Refinement (a modified impelementation of "A unified and discriminative model for query refinement"
Guo, Jiafeng, et al. "A unified and discriminative model for query refinement." Proceedings of the 31st annual international ACM SIGIR conference on Research and development in information retrieval. 2008.
https://dl.acm.org/doi/abs/10.1145/1390334.1390400)


## DISCRIMINATIVE‬‬ ‫‪MODEL‬‬ ‫‪FOR‬‬ ‫‪QUERY‬‬ ‫‪REFINEMENT‬‬

### Examples
![image](https://user-images.githubusercontent.com/63575641/140634414-4273513b-4655-4bd5-82ff-9dc51987d9e1.png)

### Tasks and Operations
![image](https://user-images.githubusercontent.com/63575641/140634423-2d27e679-6c19-4f7c-b99a-b002d0f04796.png)

### Features
Lexicon-based feature representing whether a query word
or a refined query word is in a lexicon or a stopword
list.

1. Position-based feature representing whether a query word
is at the beginning, middle, or end of the query.

1. Word-based feature representing whether a query word
consists of digit, alphabet, or a mix of the two, and
whether the length of a query word is in a certain
range.

1. Corpus-based feature representing whether the frequency
of a query word or a refined query word in the corpus
exceeds a certain threshold.

1. Query-based feature representing whether the query is a
single word query or multi-word query

### Simple Model

![image](https://user-images.githubusercontent.com/63575641/140634647-4f91b83b-dc97-4013-85e1-640f57e4bdfb.png)

### Model

![image](https://user-images.githubusercontent.com/63575641/140634663-e746a48a-33f9-4eed-ab3f-4f5437e4366f.png)


* Maximum Likelihood Estimation is used in learning and the Viterbi Algorithm in predictio. 

![image](https://user-images.githubusercontent.com/63575641/140634705-8d0bf48b-9910-4b75-8f44-10cb18166b1d.png)


You can find required and more datasets(2gram64k10m.arpa) in below links:

1. https://mega.nz/folder/8AsAwAoZ#RXQQa1sCjAYO-xNcMRsfBw

1. https://www.keithv.com/software/giga/

