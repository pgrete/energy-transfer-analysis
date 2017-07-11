
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

Ref = pickle.load(open(RefFile,"rb"))


# In[19]:

Test = pickle.load(open(TestFile,"rb"))


# In[31]:

for Term in Ref["WW"].keys():
    for KBin in Ref["WW"][Term]['AnyToAny'].keys():
        for QBin in Ref["WW"][Term]['AnyToAny'][KBin].keys():
            np.testing.assert_allclose(Ref["WW"][Term]['AnyToAny'][KBin][QBin],
                                       Test["WW"][Term]['AnyToAny'][KBin][QBin],
                                       rtol=1e-14,atol=1e-7
                                      )            
            
print("Good to go!")        


# In[ ]:



