
##
import os.path
import math
import sys
import random
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    unk_count = 0
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
        
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            # print(word)
            # print(freqDict[word])
            if freqDict[word] < 2:

                sen[i] = UNK
                unk_count += 1  # Increment the UNK count
    print("Number of UNK words in corpus:", unk_count)
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    unk_count = 0
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
                unk_count += 1  # Increment the UNK count
    print("Number of UNK words in corpus:", unk_count)
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------


class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using linear interpolation smoothing (SmoothedBigramModelInt)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    # def generateSentencesToFile(self, numberOfSentences, filename):
    #     filePointer = open(filename, 'w+')
    #     for i in range(0,numberOfSentences):
    #         sen = self.generateSentence()
    #         prob = self.getSentenceProbability(sen)

    #         stringGenerated = str(prob) + " " + "".join(sen) 
    #         print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef

class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.UnigramDist=UnigramDist(corpus)
        
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            while(len(sen.split()) == 2):
                sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            stringGenerated = str(prob) + " " + "".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)

    def generateSentence(self):
        sentence = [start]
        sentEnd = False
        while not sentEnd:
            word=self.UnigramDist.draw()
            if word == end:
                sentEnd = True
            else:
                sentence.append(word)
        return ' '.join(sentence[1:-1]) 
        
    def getSentenceProbability(self, sen):
        probability=0.0
        sen=sen.split()
        if len(sen)!=0:
            probability=self.UnigramDist.prob(sen[0])
            for i in range(1,len(sen)):
                probability*=self.UnigramDist.prob(sen[i])
        return probability

    def getCorpusPerplexity(self, corpus):
        log_prob_sum = 0.0
        N = 0
        for sent in corpus:
            for word in sent:
                if word == start:
                    continue
                pro=self.UnigramDist.prob(word)
                log_prob_sum += math.log(pro)  
                N += 1
        avg_log_prob = log_prob_sum / N  
        perplexity = math.exp(-avg_log_prob) 
        return perplexity

class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.UnigramDist=UnigramDist_smoothed(corpus)
        
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            while(len(sen.split()) == 2):
                sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            stringGenerated = str(prob) + " " + "".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)

    def generateSentence(self):
        sentence = [start]
        sentEnd = False
        while not sentEnd:
            word=self.UnigramDist.draw()
            if word == end:
                sentEnd = True
            else:
                sentence.append(word)
        return ' '.join(sentence[1:-1])  
        
    def getSentenceProbability(self, sen):
        probability=0.0
        sen=sen.split()
        if len(sen)!=0:
            probability=self.UnigramDist.prob(sen[0])
            for i in range(1,len(sen)):
                probability*=self.UnigramDist.prob(sen[i])
        return probability

    def getCorpusPerplexity(self, corpus):
        log_prob_sum = 0.0
        N = 0
        for sent in corpus:
            for word in sent:
                if word == start:
                    continue
                pro=self.UnigramDist.prob(word)
                log_prob_sum += math.log(pro) 
                N += 1
        avg_log_prob = log_prob_sum / N  
        perplexity = math.exp(-avg_log_prob) 

        return perplexity

class BigramModel(LanguageModel):
    def __init__(self, corpus):
        self.Bgramdist=BigramDist(corpus)

    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + "".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
    def generateSentence(self):
        sentence = [start]
        sentEnd = False
        given=start
        while not sentEnd:
            word = self.Bgramdist.draw(given)
            sentence.append(word)
            given=word
            if word == end:
                sentEnd = True
        return ' '.join(sentence[1:-1])  

    def getSentenceProbability(self, sen):
        probability=0.0
        sen=sen.split()
        if len(sen)>=2:
            probability=self.Bgramdist.prob(sen[0],sen[1])
            for i in range(2,len(sen)):
                probability*=self.Bgramdist.prob(sen[i],sen[i-1])
        return probability

    def getCorpusPerplexity(self, corpus):
        log_prob_sum = 0.0
        N = 0
        for sent in corpus:
            given = start  
            first_word = True  
            for word in sent:
                if not first_word:
                    pro = self.Bgramdist.prob(word, given)
                    if pro == 0:
                        
                        return
                    log_prob_sum += math.log(pro)  
                else:
                    first_word = False 
                given = word  
                N += 1
        avg_log_prob = log_prob_sum / N  
        perplexity = math.exp(-avg_log_prob)  
        return perplexity  

class BigramModel_smoothed(LanguageModel):
    def __init__(self, corpus):
        self.Bgramdist=BigramDist_smoothed(corpus)

    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + "".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
    def generateSentence(self):
        sentence = [start]
        given = start
        while True:  
            word = self.Bgramdist.draw(given)
            sentence.append(word)
            given = word
            if word == end:
                break  
        return ' '.join(sentence[1:-1])  
        
    def getSentenceProbability(self, sen):
        probability=0.0
        sen=sen.split()
        if len(sen)>=2:
            probability=self.Bgramdist.prob(sen[0],sen[1])
            for i in range(2,len(sen)):
                probability*=self.Bgramdist.prob(sen[i],sen[i-1])
        return probability

    def getCorpusPerplexity(self, corpus):
        log_prob_sum = 0.0
        N = 0
        for sent in corpus:
            given = start  
            first_word = True  
            for word in sent:
                if not first_word:
                    pro = self.Bgramdist.prob(word, given)
                    log_prob_sum += math.log(pro)  
                else:
                    first_word = False  
                given = word  
                N += 1
        avg_log_prob = log_prob_sum / N  
        perplexity = math.exp(-avg_log_prob)  
        return perplexity

class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0

    def prob(self, word):
        return self.counts[word]/self.total
    
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
          
class UnigramDist_smoothed:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0 
    
    def prob(self, word):
        return (self.counts[word] + 1.0)/(self.total + len(self.counts)) 
    
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

class BigramDist:
    def __init__(self, corpus):
        self.bcounts = defaultdict(float)
        self.ucounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if j!=0:
                    self.bcounts[(corpus[i][j],corpus[i][j-1])]+=1
                self.ucounts[corpus[i][j]]+=1
                self.total+=1
 
    def prob(self, word, given):
       
        numerator = self.bcounts[(given, word)] 
        denominator = self.ucounts[given]
        if numerator==0 or denominator==0:
            prob=0
        else:
            prob = numerator / denominator
        return prob

    def draw(self,given):
        rand = self.ucounts[given]*random.random()
        for word,word1 in self.bcounts:
            if word1==given:
                rand-=self.bcounts[word,word1]
                if rand <= 0.0:
                    return word

class BigramDist_smoothed:
    def __init__(self, corpus):
        self.bcounts = defaultdict(float)
        self.ucounts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)

    def train(self, corpus):
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                if j != 0:
                    self.bcounts[(corpus[i][j], corpus[i][j - 1])] += 1
                self.ucounts[corpus[i][j]] += 1
                self.total += 1

    def prob(self, word, given, lambda_value=0.6):
        bigram_prob = self.bigram_prob(word, given)
        unigram_prob = self.unigram_prob(word)
        interpolated_prob = lambda_value * bigram_prob + (1 - lambda_value) * unigram_prob
        return interpolated_prob

    def bigram_prob(self, word, given):
        numerator = self.bcounts[(word, given)]
        denominator = self.ucounts[given]
        if numerator == 0 or denominator == 0:
            return 0
        else:
            return numerator / denominator

    def unigram_prob(self, word):
        if word in self.ucounts:
            return self.ucounts[word] / self.total
        else:
            return 0

    def draw(self, given):
        rand = self.ucounts[given]*random.random()
        for word,word1 in self.bcounts:
            if word1==given:
                rand-=self.bcounts[word,word1]
                if rand <= 0.0:
                    return word    

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":

    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    vocab = set() 
    for sentenceList in trainCorpus:
        vocab.update(sentenceList)
    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

# Unigram Language Model

    unigramDist = UnigramDist(trainCorpus)
    unigram = UnigramModel(trainCorpus)
    unigram.generateSentencesToFile(20, "unigram_output.txt")
    trainPerp = unigram.getCorpusPerplexity(trainCorpus)
    posPerp = unigram.getCorpusPerplexity(posTestCorpus)
    negPerp = unigram.getCorpusPerplexity(negTestCorpus)   
    print ("Perplexity of Unigram on positive training corpus:    "+ str(trainPerp))
    print ("Perplexity of Unigram on positive review test corpus: "+ str(posPerp))
    print ("Perplexity of Unigram on negative review test corpus: "+ str(negPerp))

# Unigram Language Model with Laplace smoothing

    unigramDist_smooth = UnigramDist_smoothed(trainCorpus)
    unigram_smoothed = SmoothedUnigramModel(trainCorpus)
    unigram_smoothed.generateSentencesToFile(20, "smoothed_unigram_output.txt")
    trainPerp = unigram_smoothed.getCorpusPerplexity(trainCorpus)
    posPerp = unigram_smoothed.getCorpusPerplexity(posTestCorpus)
    negPerp = unigram_smoothed.getCorpusPerplexity(negTestCorpus)   
    print ("Perplexity of Unigram on positive training corpus:    SMOOTHED "+ str(trainPerp))
    print ("Perplexity of Unigram on positive review test corpus: SMOOTHED "+ str(posPerp))
    print ("Perplexity of Unigram on negative review test corpus: SMOOTHED "+ str(negPerp))

# Bigram Language Model

    bigramDist = BigramDist(trainCorpus)
    bigram = BigramModel(trainCorpus)
    bigram.generateSentencesToFile(20,"bigram_output.txt")
    trainPerp = bigram.getCorpusPerplexity(trainCorpus)
    posPerp = bigram.getCorpusPerplexity(posTestCorpus)
    negPerp = bigram.getCorpusPerplexity(negTestCorpus)   
    print ("Perplexity of Bigram on positive training corpus:    "+ str(trainPerp))
    print ("Perplexity of Bigram on positive review test corpus: "+ str(posPerp))
    print ("Perplexity of Bigram on negative review test corpus: "+ str(negPerp))

# Bigram Language Model with Linear Interpolation Smoothing

    bigramDist_smoothed = BigramDist_smoothed(trainCorpus)
    bigram_s = BigramModel_smoothed(trainCorpus)
    bigram_s.generateSentencesToFile(20,"bigram_smoothed_output.txt")
    trainPerp = bigram_s.getCorpusPerplexity(trainCorpus)
    posPerp = bigram_s.getCorpusPerplexity(posTestCorpus)
    negPerp = bigram_s.getCorpusPerplexity(negTestCorpus)   
    print ("Perplexity of Bigram on positive training corpus SMOOTHED:    "+ str(trainPerp))
    print ("Perplexity of Bigram on positive review test corpus: SMOOTHED "+ str(posPerp))
    print ("Perplexity of Bigram on negative review test corpus: SMOOTHED "+ str(negPerp))

# Generally, the perplexity of the models tends to be higher (indicating lower model performance) 
# on the positive review test corpus compared to the negative review test corpus. 
# This suggests that the language models are better able to predict words in the context of negative reviews compared to positive reviews. 
    



# Questions:


#1. When generating sentences with the unigram model, what controls the length of the generated 
#sentences? How does this differ from the sentences produced by the bigram models?
    
    # the draw function controls the length of the generated sentence!
    # until the end of sentence token occurs by chance

    # Same goes for the bigram model with a slight difference where 
    # next word is randomly picked based on the previously occured word

# 2. Consider the probability of the generated sentences according to your models. Do your models assign
# drastically different probabilities to the different sets of sentences? Why do you think that is?
    
    # As the next word is randomly picked from the distribution
    # the sentence length vary as well as the words selected
    # where sentences having multiple UNK tokens will have 
    # a higher expected probability than others due to their
    # higher occurance 

# 3. Generate additional sentences using your bigram and smoothed bigram models. In your opinion, which
# model produces better / more realistic sentences?
    
    # The sentences generated by the smoothed bigram model seem
    # to have a better flow, with more coherent structure and language use. 
    # On the other hand, the sentences generated by the regular bigram model 
    # often lack coherence and grammatical correctness.

# 4. For each of the four models, which test corpus has a higher perplexity? Why? Make sure to include the 
# perplexity values in the answer.

    # Perplexity of Unigram on positive review test corpus: 628.6696007318275
    # Perplexity of Unigram on negative review test corpus: 612.2628559647762

    # Perplexity of Unigram on positive review test corpus: SMOOTHED 631.5562305193403
    # Perplexity of Unigram on negative review test corpus: SMOOTHED 615.3626467738206

    # Perplexity of Bigram on positive review test corpus: 564.81958292093
    # Perplexity of Bigram on negative review test corpus: 546.595827649828

    # Perplexity of Bigram on positive review test corpus: SMOOTHED 195.25301444629784
    # Perplexity of Bigram on negative review test corpus: SMOOTHED 200.40591482183225

    # Generally speaking, the positive review test corpus has a higher perplexity in 
    # simple unigram, smoothed unigram and unsmoothed bigram model. The reason is that
    # the number of unk words in training corpus is high adding higher probability to 
    # UNK words collectively, whereas, the unk words in positive test data are less as
    # compared to the negative review test corpus hence, having an overall higher perplexity
    
    # Here are the UNK words count

    # Number of UNK words in training corpus: 15689
    # Number of UNK words in positive riview test corpus: 1118
    # Number of UNK words in positive riview test corpus: 1166