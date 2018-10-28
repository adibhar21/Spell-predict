# -*- coding: utf-8 -*-


import sys 
from nltk.metrics import edit_distance
import pickle
import difflib as dl

try:
    wrong_word = sys.argv[1]

except IndexError:
    sys.exit('ERROR : No word supplied as input')

class Spell_correction_singleton:
      def __init__(self,count_unique,count_unigram,count_bigram):
          self.a = count_unique
          self.b = count_unigram
          self.c = count_bigram

#from Panini import Complex
with open('address', 'rb') as f:
    data = pickle.load(f)

print('Executing...')

candidate_list_1 = [] # candidate correction list with edit distance 1
candidate_list_2 = [] # candidate correction list with edit distance 2


for w in data.a: # retreiving candidates using nltk library of levenshtein edits
    candidate_distance = edit_distance(wrong_word,w, substitution_cost = 1, transpositions= True)
    if(candidate_distance==1):
        candidate_list_1.append(w)
    if(candidate_distance==2):
        candidate_list_2.append(w)


# using alphabets for reference in confusion matrix...y-axis.
dict_alph = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5,
             'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
             'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17,
             's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23,
             'y': 24, 'z': 25, '@': 26}

# using the confusion matrix in kernighan for four operations

add_table = [['a',15,1,14,7,10,0,1,1,33,1,4,31,2,39,12,4,3,28,134,7,28,0,1,1,4,1],
['b',3,11,0,0,7,0,1,0,50,0,0,15,0,1,1,0,0,5,16,0,0,3,0,0,0,0],
['c',19,0,54,1,13,0,0,18,50,0,3,1,1,1,7,1,0,7,25,7,8,4,0,1,0,0],
['d',18,0,3,17,14,2,0,0,9,0,0,6,1,9,13,0,0,6,119,0,0,0,0,0,5,0],
['e',39,2,8,76,147,2,0,1,4,0,3,4,6,27,5,1,0,83,417,6,4,1,10,2,8,0],
['f',1,0,0,0,2,27,1,0,12,0,0,10,0,0,0,0,0,5,23,0,1,0,0,0,1,0],
['g',8,0,0,0,5,1,5,12,8,0,0,2,0,1,1,0,1,5,69,2,3,0,1,0,0,0],
['h',4,1,0,1,24,0,10,18,17,2,0,1,0,1,4,0,0,16,24,22,1,0,5,0,3,0],
['i',10,3,13,13,25,0,1,1,69,2,1,17,11,33,27,1,0,9,30,29,11,0,0,1,0,1],
['j',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
['k',2,4,0,1,9,0,0,1,1,0,1,1,0,0,2,1,0,0,95,0,1,0,0,0,4,0],
['l',3,1,0,1,38,0,0,0,79,0,2,128,1,0,7,0,0,0,97,7,3,1,0,0,2,0],
['m',11,1,1,0,17,0,0,1,6,0,1,0,102,44,7,2,0,0,47,1,2,0,1,0,0,0],
['n',15,5,7,13,52,4,17,0,34,0,1,1,26,99,12,0,0,2,156,53,1,1,0,0,1,0],
['o',14,1,1,3,7,2,1,0,28,1,0,6,3,13,64,30,0,16,59,4,19,1,0,0,1,1],
['p',23,0,1,1,10,0,0,20,3,0,0,2,0,0,26,70,0,29,52,9,1,1,1,0,0,0],
['q',0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
['r',15,2,1,0,89,1,1,2,64,0,0,5,9,7,10,0,0,132,273,29,7,0,1,0,10,0],
['s',13,1,7,20,41,0,1,50,101,0,2,2,10,7,3,1,0,1,205,49,7,0,1,0,7,0],
['t',39,0,0,3,65,1,10,24,59,1,0,6,3,1,23,1,0,54,264,183,11,0,5,0,6,0],
['u',15,0,3,0,9,0,0,1,24,1,1,3,3,9,1,3,0,49,19,27,26,0,0,2,3,0],
['v',0,2,0,0,36,0,0,0,10,0,0,1,0,1,0,1,0,0,0,0,1,5,1,0,0,0],
['w',0,0,0,1,10,0,0,1,1,0,1,1,0,2,0,0,1,1,8,0,2,0,4,0,0,0],
['x',0,0,18,0,1,0,0,6,1,0,0,0,1,0,3,0,0,0,2,0,0,0,0,1,0,0],
['y',5,1,2,0,3,0,0,0,2,0,0,1,1,6,0,0,0,1,33,1,13,0,1,0,2,0],
['z',2,0,0,0,5,1,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4],
['@',46,8,9,8,26,11,14,3,5,1,17,5,6,2,2,10,0,6,23,2,11,1,2,1,1,2]]

del_table = [['a',0,7,58,21,3,5,18,8,61,0,4,43,5,53,0,9,0,98,28,53,62,1,0,0,2,0],
['b',2,2,1,0,22,0,0,0,183,0,0,26,0,0,2,0,0,6,17,0,6,1,0,0,0,0],
['c',37,0,70,0,63,0,0,24,320,0,9,17,0,0,33,0,0,46,6,54,17,0,0,0,1,0],
['d',12,0,7,25,45,0,10,0,62,1,1,8,4,3,3,0,0,11,1,0,3,2,0,0,6,0],
['e',80,1,50,74,89,3,1,1,6,0,0,32,9,76,19,9,1,237,223,34,8,2,1,7,1,0],
['f',4,0,0,0,13,46,0,0,79,0,0,12,0,0,4,0,0,11,0,8,1,0,0,0,1,0],
['g',25,0,0,2,83,1,37,25,39,0,0,3,0,29,4,0,0,52,7,1,22,0,0,0,1,0],
['h',15,12,1,3,20,0,0,25,24,0,0,7,1,9,22,0,0,15,1,26,0,0,1,0,1,0],
['i',26,1,60,26,23,1,9,0,1,0,0,38,14,82,41,7,0,16,71,64,1,1,0,0,1,7],
['j',0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0],
['k',4,0,0,1,15,1,8,1,5,0,1,3,0,17,0,0,0,1,5,0,0,0,1,0,0,0],
['l',24,0,1,6,48,0,0,0,217,0,0,211,2,0,29,0,0,2,12,7,3,2,0,0,11,0],
['m',15,10,0,0,33,0,0,1,42,0,0,0,180,7,7,31,0,0,9,0,4,0,0,0,0,0],
['n',21,0,42,71,68,1,160,0,191,0,0,0,17,144,21,0,0,0,127,87,43,1,1,0,2,0],
['o',11,4,3,6,8,0,5,0,4,1,0,13,9,70,26,20,0,98,20,13,47,2,5,0,1,0],
['p',25,0,0,0,22,0,0,12,15,0,0,28,1,0,30,93,0,58,1,18,2,0,0,0,0,0],
['q',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0],
['r',63,4,12,19,188,0,11,5,132,0,3,33,7,157,21,2,0,277,103,68,0,10,1,0,27,0],
['s',16,0,27,0,74,1,0,18,231,0,0,2,1,0,30,30,0,4,265,124,21,0,0,0,1,0],
['t',24,1,2,0,76,1,7,49,427,0,0,31,3,3,11,1,0,203,5,137,14,0,4,0,2,0],
['u',26,6,9,10,15,0,1,0,28,0,0,39,2,111,1,0,0,129,31,66,0,0,0,0,1,0],
['v',9,0,0,0,58,0,0,0,31,0,0,0,0,0,2,0,0,1,0,0,0,0,0,0,1,0],
['w',40,0,0,1,11,1,0,11,15,0,0,1,0,2,2,0,0,2,24,0,0,0,0,0,0,0],
['x',1,0,17,0,3,0,0,1,0,0,0,0,0,0,0,6,0,0,0,5,0,0,0,0,1,0],
['y',2,1,34,0,2,0,1,0,1,0,0,1,2,1,1,1,0,0,17,1,0,0,1,0,0,0],
['z',1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
['@',20,14,41,31,20,20,7,6,20,3,6,22,16,5,5,17,0,28,26,6,2,1,24,0,0,2]]

sub_table = [['a',0,0,7,1,342,0,0,2,118,0,1,0,0,3,76,0,0,1,35,9,9,0,1,0,5,0],
['b',0,0,9,9,2,2,3,1,0,0,0,5,11,5,0,10,0,0,2,1,0,0,8,0,0,0],
['c',6,5,0,16,0,9,5,0,0,0,1,0,7,9,1,10,2,5,39,40,1,3,7,1,1,0],
['d',1,10,13,0,12,0,5,5,0,0,2,3,7,3,0,1,0,43,30,22,0,0,4,0,2,0],
['e',388,0,3,11,0,2,2,0,89,0,0,3,0,5,93,0,0,14,12,6,15,0,1,0,18,0],
['f',0,15,0,3,1,0,5,2,0,0,0,3,4,1,0,0,0,6,4,12,0,0,2,0,0,0],
['g',4,1,11,11,9,2,0,0,0,1,1,3,0,0,2,1,3,5,13,21,0,0,1,0,3,0],
['h',1,8,0,3,0,0,0,0,0,0,2,0,12,14,2,3,0,3,1,11,0,0,2,0,0,0],
['i',103,0,0,0,146,0,1,0,0,0,0,6,0,0,49,0,0,0,2,1,47,0,2,1,15,0],
['j',0,1,1,9,0,0,1,0,0,0,0,2,1,0,0,0,0,0,5,0,0,0,0,0,0,0],
['k',1,2,8,4,1,1,2,5,0,0,0,0,5,0,2,0,0,0,6,0,0,0,.4,0,0,3],
['l',2,10,1,4,0,4,5,6,13,0,1,0,0,14,2,5,0,11,10,2,0,0,0,0,0,0],
['m',1,3,7,8,0,2,0,6,0,0,4,4,0,180,0,6,0,0,9,15,13,3,2,2,3,0],
['n',2,7,6,5,3,0,1,19,1,0,4,35,78,0,0,7,0,28,5,7,0,0,1,2,0,2],
['o',91,1,1,3,116,0,0,0,25,0,2,0,0,0,0,14,0,2,4,14,39,0,0,0,18,0],
['p',0,11,1,2,0,6,5,0,2,9,0,2,7,6,15,0,0,1,3,6,0,4,1,0,0,0],
['q',0,0,1,0,0,0,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
['r',0,14,0,30,12,2,2,8,2,0,5,8,4,20,1,14,0,0,12,22,4,0,0,1,0,0],
['s',11,8,27,33,35,4,0,1,0,1,0,27,0,6,1,7,0,14,0,15,0,0,5,3,20,1],
['t',3,4,9,42,7,5,19,5,0,1,0,14,9,5,5,6,0,11,37,0,0,2,19,0,7,6],
['u',20,0,0,0,44,0,0,0,64,0,0,0,0,2,43,0,0,4,0,0,0,0,2,0,8,0],
['v',0,0,7,0,0,3,0,0,0,0,0,1,0,0,1,0,0,0,8,3,0,0,0,0,0,0],
['w',2,2,1,0,1,0,0,2,0,0,1,0,0,0,0,7,0,6,3,3,1,0,0,0,0,0],
['x',0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0],
['y',0,0,2,0,15,0,1,7,15,0,0,0,2,0,6,1,0,7,36,8,5,0,0,1,0,0],
['z',0,0,0,7,0,0,0,0,0,0,0,7,5,0,0,0,0,2,21,3,0,0,0,0,3,0]]

transpose_table = [['a',0,0,2,1,1,0,0,0,19,0,1,14,4,25,10,3,0,27,3,5,31,0,0,0,0,0],
['b',0,0,0,0,2,0,0,0,0,0,0,1,1,0,2,0,0,0,2,0,0,0,0,0,0,0],
['c',0,0,0,0,1,0,0,1,85,0,0,15,0,0,13,0,0,0,3,0,7,0,0,0,0,0],
['d',0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0],
['e',1,0,4,5,0,0,0,0,60,0,0,21,6,16,11,2,0,29,5,0,85,0,0,0,2,0],
['f',0,0,0,0,0,0,0,0,12,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
['g',4,0,0,0,2,0,0,0,0,0,0,1,0,15,0,0,0,3,0,0,3,0,0,0,0,0],
['h',12,0,0,0,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0],
['i',15,8,31,3,66,1,3,0,0,0,0,9,0,5,11,0,1,13,42,35,0,6,0,0,0,3],
['j',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
['k',0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
['l',11,0,0,12,20,0,1,0,4,0,0,0,0,0,1,3,0,0,1,1,3,9,0,0,7,0],
['m',9,0,0,0,20,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,4,0,0,0,0,0],
['n',15,0,6,2,12,0,8,0,1,0,0,0,3,0,0,0,0,0,6,4,0,0,0,0,0,0],
['o',5,0,2,0,4,0,0,0,5,0,0,1,0,5,0,1,0,11,1,1,0,0,7,1,0,0],
['p',17,0,0,0,4,0,0,1,0,0,0,0,0,0,1,0,0,5,3,6,0,0,0,0,0,0],
['q',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
['r',12,0,0,0,24,0,3,0,14,0,2,2,0,7,30,1,0,0,0,2,10,0,0,0,2,0],
['s',4,0,0,0,9,0,0,5,15,0,0,5,2,0,1,22,0,0,0,1,3,0,0,0,16,0],
['t',4,0,3,0,4,0,0,21,49,0,0,4,0,0,3,0,0,5,0,0,11,0,2,0,0,0],
['u',22,0,5,1,1,0,2,0,2,0,0,2,1,0,20,2,0,11,11,2,0,0,0,0,0,0],
['v',0,0,0,0,1,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
['w',0,0,0,0,0,0,0,4,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,8,0],
['x',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
['y',0,1,2,0,0,0,1,0,0,0,0,3,0,0,0,2,0,1,10,0,0,0,0,0,0,0],
['z',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]


# using dict comprehension to convert nested lists of confusion matrix to dicts
dict_del = { i[0]:i[1:] for i in del_table}
dict_add = { i[0]:i[1:] for i in add_table}
dict_sub = { i[0]:i[1:] for i in sub_table}
dict_tran = { i[0]:i[1:] for i in transpose_table}
changes = []
# changes noted or operations that can be performed for these four operations -insert,delete,substitute and transpose(insert and delete)


# each of these uses indexes of [op_code] function in difflib library of python
def del_prob(index):
        deleted = candidate_list_1[index][changes[0][3]:changes[0][4]] # correct word and their index that was inserted
        prior = candidate_list_1[index][changes[0][3]-1:changes[0][4]-1] # noisy channel uses prior as reference
        if prior != "":
            merge = prior+deleted
        if prior == "":
            prior1 = "<"
            prior = "@"
            merge = prior1+deleted
        channel_prob = dict_del[prior][dict_alph[deleted]]/data.c[merge] #channel model probability
        word_prob = data.a[candidate_list_1[index]]/data.b['<']  #correct word divided by number of words probability
        total_prob = (dict_del[prior][dict_alph[deleted]]/data.c[merge] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000) # total probability
        correct_letter = deleted
        error_letter = "_"
        XW = prior+'|'+prior+correct_letter
        list_probability.append([total_prob,candidate_list_1[index],"insert",correct_letter,error_letter,XW,channel_prob,word_prob]) # list for printing a table
        

def insert_prob(index):
        inserted = wrong_word[changes[0][1]:changes[0][2]] # wrong word and their index that was deleted
        prior = candidate_list_1[index][changes[0][1]-1:changes[0][2]-1] # noisy channel uses prior as reference ...correct word
        if prior == inserted and prior !="":
                 prior2 = candidate_list_1[index][changes[0][1]-2:changes[0][2]-2]
                 channel_prob2 = dict_add[prior2][dict_alph[inserted]]/data.b[prior2] #channel model probability
                 word_prob = data.a[candidate_list_1[index]]/data.b['<'] #correct word divided by number of words probability
                 total_prob2 = (dict_add[prior2][dict_alph[inserted]]/data.b[prior2] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000) # total probability
                 correct_letter = "_"
                 error_letter = inserted
                 XW2 = prior2+error_letter+'|'+prior2
                 list_probability.append([total_prob2,candidate_list_1[index],"delete",correct_letter,error_letter,XW2,channel_prob2,word_prob])
                 channel_prob = dict_add[prior][dict_alph[inserted]]/data.b[prior]
                 total_prob = (dict_add[prior][dict_alph[inserted]]/data.b[prior] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000)
                 XW = prior+error_letter+'|'+prior
                 list_probability.append([total_prob,candidate_list_1[index],"delete",correct_letter,error_letter,XW,channel_prob,word_prob])
                 return
        if prior != "":
            merge = prior
        if prior == "":
            prior1 = "<"
            prior = "@"
            merge = prior1
        channel_prob = dict_add[prior][dict_alph[inserted]]/data.b[merge]
        word_prob = data.a[candidate_list_1[index]]/data.b['<']
        total_prob = (dict_add[prior][dict_alph[inserted]]/data.b[merge] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000)
        correct_letter = "_"
        error_letter = inserted
        XW = prior+error_letter+'|'+prior
        list_probability.append([total_prob,candidate_list_1[index],"delete",correct_letter,error_letter,XW,channel_prob,word_prob])        
        

def subs_prob(index): # substitution using op_code values and indexing for correct position of letter(s) 
        substituted = candidate_list_1[index][changes[0][1]:changes[0][2]]
        prior = wrong_word[changes[0][1]:changes[0][2]]
        channel_prob = dict_sub[prior][dict_alph[substituted]]/data.b[substituted]
        word_prob = data.a[candidate_list_1[index]]/data.b['<']
        total_prob = (dict_sub[prior][dict_alph[substituted]]/data.b[substituted] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000)
        correct_letter = substituted
        error_letter = prior
        XW = prior+'|'+substituted
        list_probability.append([total_prob,candidate_list_1[index],"replace",correct_letter,error_letter,XW,channel_prob,word_prob])

# special case for substitution using op_code values
def special_case_prob(index):
    substituted = candidate_list_1[index][changes[0][3]]
    prior = wrong_word[changes[0][3]]
    channel_prob = dict_sub[prior][dict_alph[substituted]]/data.b[substituted]
    word_prob = data.a[candidate_list_1[index]]/data.b['<']
    total_prob = (dict_sub[prior][dict_alph[substituted]]/data.b[substituted] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000)
    correct_letter = candidate_list_1[index][changes[0][3]]
    error_letter = wrong_word[changes[0][3]]
    XW = wrong_word[changes[0][3]]+'|'+candidate_list_1[index][changes[0][3]]
    list_probability.append([total_prob,candidate_list_1[index],"replace",correct_letter,error_letter,XW,channel_prob,word_prob])

# transpose operation using op_code indexing
def transp_prob(index):
        transpose_first = candidate_list_1[index][changes[0][3]:changes[0][4]]
        transpose_second =  candidate_list_1[index][changes[1][1]:changes[1][2]]
        if transpose_first!= "" and transpose_second!="":
            merge = transpose_first+transpose_second
        channel_prob = dict_tran[transpose_first][dict_alph[transpose_second]]/data.c[merge]
        word_prob = data.a[candidate_list_1[index]]/data.b['<']
        total_prob = (dict_tran[transpose_first][dict_alph[transpose_second]]/data.c[merge] * (data.a[candidate_list_1[index]]/data.b['<']) * 1000000000)
        correct_letters = merge
        error_letters = transpose_second + transpose_first
        XW = error_letters + '|' + correct_letters
        list_probability.append([total_prob,candidate_list_1[index],"transpose",correct_letters,error_letters,XW,channel_prob,word_prob])

list_probability = [] #list of all candidates with their complete set of probabilities

# processing each candidate word and routing them to particular operation for computing probability
for i in range(len(candidate_list_1)): 
    seq = dl.SequenceMatcher(None,wrong_word,candidate_list_1[i]) # sequence matching function of difflib in python
    p = seq.get_opcodes() # using op_code to understand how a incorrect word could be turned into correct word
    changes = [] #change or operations recognized by the op_code function
    changes = list(filter(lambda x: not x[0].startswith('equal'), p)) # obtaining only operations that recognizes changes in a word
    
    if (len(changes)>1): #insertion and deletion is what fetches us a transpose, with op_code greater than 1..it's definitely a transpose 
        if len(list(filter(lambda x: x[0]!=x[1], list(zip(wrong_word,candidate_list_1[i])) )))==1: #acress --> acres
            special_case_prob(i)
        if len(list(filter(lambda x: x[0]!=x[1], list(zip(wrong_word,candidate_list_1[i])) )))>1:
            transp_prob(i)
    if (changes[0][0]=="insert" and len(changes)==1): #calling each function for computing probability
        del_prob(i)
    if (changes[0][0]=="delete" and len(changes)==1):
        insert_prob(i)
    if (changes[0][0]=="replace" and len(changes)==1):
        subs_prob(i)

result_list = sorted(list_probability,reverse=True) # sorting by descending order of candidate word probabilities
print('\n\n')
print('Mistyped word: ',wrong_word)
print('\n')
print('{:<8}'.format("Candidate"),'{:>8}'.format("Correct"),'{:>8}'.format("error"),'{:>7}'.format("X|W"),'{:>12}'.format("P(X|W)"),'{:>15}'.format("P(Word)"),'{:>22}'.format("10^9*P(x|w)P(w)"))
print('\n')
for e1,e2,e3,e4,e5,e6,e7,e8 in result_list:
#    print("{:>15}{:6f} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(e1,e2,e3,e4,e5,e6,e7,e8))
        print("{:<8} {:>5} {:>10} {:>10} {:>15,.10f} {:>15,.10f} {:>15,.6f}".format(e2,e4,e5,e6,e7,e8,e1))
print('\n')

#printing candidates with 2 edit distances
if len(candidate_list_2)>1:
    print('\tOther candidate corrections with 2 edit distances: \n\n',candidate_list_2)

# =============================================================================
# if user deleted --> def del
# opcode --> insert
# table --> delete
# =============================================================================
    
# =============================================================================
# if user inserted --> def insert
# opcode --> delete
# table --> add
# =============================================================================
    
    
  
