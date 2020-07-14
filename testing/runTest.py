
# coding: utf-8

# In[2]:

import numpy as np


# In[3]:

import pickle
import sys


# In[ ]:

RefFile = sys.argv[1]
TestFile = sys.argv[2]


# In[18]:
if sys.version_info[0] < 3:
    Ref = pickle.load(open(RefFile,"rb"))
    Test = pickle.load(open(TestFile,"rb"))
else:
    Ref = pickle.load(open(RefFile,"rb"),encoding='latin1')
    Test = pickle.load(open(TestFile,"rb"),encoding='latin1')


# In[31]:

test_terms = list(Test["WW"].keys())

for Term in Ref["WW"].keys():
    if Term not in Test["WW"].keys():
        print('missing',Term,'in test set.')
        continue
    else:
        print("Checking...",Term)
        test_terms.remove(Term)

    for KBin in Ref["WW"][Term]['AnyToAny'].keys():
        for QBin in Ref["WW"][Term]['AnyToAny'][KBin].keys():
            np.testing.assert_allclose(Ref["WW"][Term]['AnyToAny'][KBin][QBin],
                                       Test["WW"][Term]['AnyToAny'][KBin][QBin],
                                       rtol=1e-14,atol=1e-7
                                      )            
if len(test_terms) > 0:
    print("Got extra terms in test set:",test_terms)
print("Good to go!")        


# In[ ]:



