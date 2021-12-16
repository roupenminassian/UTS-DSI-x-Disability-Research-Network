# These lines of code allow for manually chunking sentences or paragraphs of extracted information to a list
# The string that is defined for 'A' will be manually updated each time the previous string has been added to GovList
# This means that we are incrementaly adding sentences and paragraphs to this list
# This particular formatting is beneficial for BM25, to allow us to retieve the most relevant sentences or paragraphs related to our query
#The pickle library also allows us to save this list to be used for later.

import pickle #import pickle library

GovList = [] #Setup an empty list

A = "[Insert Text Here]" #Define A to a string. This will change everytime the previous string is added to the list.

GovList.append(A) #Append string to list

len(GovList) #Check the length of the list

with open("test.txt", "wb") as fp:   #Pickling the list in order to save to a file
...   pickle.dump(GovList, fp)

with open("/content/drive/MyDrive/test.txt","rb") as fp:# Unpickling the list to use in the notebook
  b = pickle.load(fp)
