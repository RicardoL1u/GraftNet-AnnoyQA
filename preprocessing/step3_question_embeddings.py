"""Script to compute question embeddings for given questions."""

questions_file = "scratch/webqsp_processed.json"
embeddings_file = "glove"
output_file = "scratch/webqsp_embeddings.pkl"
dim = 300

import pickle as pkl
import numpy as np
import json
from tqdm import tqdm

def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

word_to_question = {}
question_lens = {}
def _add_word(word, v):
    if word not in word_to_question: word_to_question[word] = []
    word_to_question[word].append(v)
    if v not in question_lens: question_lens[v] = 0
    question_lens[v] += 1

with open(questions_file) as f:
    questions = json.load(f)
    for ii, question in enumerate(questions):
        qId, question_text = question["QuestionId"], question["QuestionKeywords"]
        for word in question_text.split():
            _add_word(word, qId)

question_emb = {r: np.zeros((dim,)) for r in question_lens}
with open(embeddings_file) as f:
    for line in tqdm(f.readlines()):
        cnt = line.count(' ')
        br = find_nth(line,' ',cnt-300+1)
        word = line[:br].strip()
        vec = line[br:].strip()
        if word in word_to_question:
            for qid in word_to_question[word]:
                question_emb[qid] += np.array([float(vv) for vv in vec.split()])

for question in question_emb:
    question_emb[question] = question_emb[question] / question_lens[question]

pkl.dump(question_emb, open(output_file, "wb"))
