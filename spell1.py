
import re
from collections import defaultdict,Counter
import time
import pickle

no_of_lines = 0
count_words=0
try:
    file = "address"
except FileNotFoundError:
    print('Correct path of file was not provided')
#count_unique_words = Counter()
#count_unigrams = Counter()
#count_bigrams = Counter()

count_unique_words = defaultdict(int) # dicts used for counting
count_unigrams = defaultdict(int)
count_bigrams = defaultdict(int)

string = "" #this is set zero before it becomes <word> after each line is processed
perf_start = time.perf_counter() #time counters
process_start = time.process_time()

with open(file) as f:
    for line in f:
        list_angular = [] #list of words with angular brackets
        no_of_lines+=1
        modified_line = re.sub(r'^\w+',' ',line) # applying regex for each line in text file and retrieving only words == [a-zA-Z]
        modified_line = re.sub(r'[^a-z]', ' ',modified_line, flags=re.IGNORECASE) # taking out any character which is not a ASCII letter
        modified_lower = modified_line.lower() # converting to lower case
        modified_line_lower = modified_lower.split() # tokenizing, this has better performance than --> tokenize = nltk.word_tokenize(modified_line.lower())
        for w in range(len(modified_line_lower)):
            list_angular.append('<'+modified_line_lower[w]+'>')   #placing angular brackets before the beginning and end of each word
        string = ' '.join(list_angular) # making it a string from list of strings
        for w in modified_line_lower:
            count_words+=1            # counting number of words in the corpus
        for w in modified_line_lower:
            count_unique_words[w]+=1  # unique words in corpus
        for w in re.findall(r'(?=([a-z\<\>]{1}))',string):
            count_unigrams[w]+=1 # unigram count of each word
        for w in re.findall(r'(?=([a-z\<\>]{2}))',string):
            count_bigrams[w]+=1   #bigram count of each word

perf_stop = time.perf_counter() # time counter end
process_stop = time.process_time()
total_time = perf_stop - perf_start
total_cpu_time = process_stop - process_start
print('\n\n')
print('start at elapsed time:{:>5.2f}, cpu time:{:>10}'.format(perf_start,process_start))
print('finish reading at elapsed time:{:>10.2f}, cpu time:{:>10}'.format(perf_stop,process_stop))
print('total elapsed time:{:>5.2f}, cpu time:{:>10}'.format(total_time,total_cpu_time))
print('\n\n')

#class Singleton1:
#     _shared_state = {}
#     def __init__(self):
#        self.__dict__ = self._shared_state
#        
#class Singleton2(Singleton1):
#    def __init__(self, arg):
#        Singleton1.__init__(self)
#        self.val = arg        
#    def __dict__(self):
#        return self.val
class Spell_correction_singleton: # singleton class ...the above has only one instance, this has three
      def __init__(self,count_unique,count_unigram,count_bigram):
          self.a = count_unique
          self.b = count_unigram
          self.c = count_bigram
          
object_1 = Spell_correction_singleton(count_unique_words,count_unigrams,count_bigrams)
 #object_1 is the only object of singleton class
   

output = open('pickled_data.dat', 'wb') #pickling the data structures
pickle.dump(object_1, output,-1)
output.close()

print("Number of words: ",count_words)
print("Number of types (distinct words): ",len(count_unique_words))
print('\n')
print('{:<8}'.format("Unigrams"),'{:>8}'.format("Count"))
for (key,value) in sorted(count_unigrams.items()):
    print("{:<8} {:>10}".format(key,value))
print('\n')

print('{:<8}'.format("Bigrams"),'{:>8}'.format("Count"))
for (key,value) in sorted(count_bigrams.items()):
    print("{:<8} {:>10}".format(key,value))
print('\n')

print("10 most frequent words: ")
print('\n')
print('{:<8}'.format("Word"),'{:>8}'.format("Count"))
for (key,value) in (Counter(count_unique_words).most_common(10)):
    print("{:<8} {:>10}".format(key,value))
print('\n')
