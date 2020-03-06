#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
from pyspark import SparkContext
import numpy as np
import heapq as hq
from collections import Counter 

sc = SparkContext()
# load up all of the  26,754 documents in the corpus
corpus = sc.textFile ("s3://risamyersbucket/A4/pubmed.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id=' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id=') + 3 : x.index('> ')], x[x.index('> ') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map(lambda x:(topWords[x][0],x))


# finally, print out some of the dictionary, just for debugging
dict=dictionary.collect()
    
def task_1(key):
    # create RDD for the document with index as a key like Wounds/23778438
    Word_CountPair = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1] if x[0] == key))
    # reduce it and get the count for each word here key is the word
    count = Word_CountPair.reduceByKey(lambda a, b, : a + b)
    # then join the dictionary with the same word as key
    Dict_WordPair = dictionary.leftOuterJoin(count)
    # solve sparse of the data
    Id_WordCountPair = Dict_WordPair.map(lambda x: (x[1][0], 0) if x[1][1] == None else (x[1][0], x[1][1]))
    ans = Id_WordCountPair.sortBy(lambda x: x[0])
    ans = ans.map(lambda x: x[1])
    ans = np.array(ans.collect())
    return ans

ans1_1=task_1('Wounds/23778438')
ans1_2=task_1('ParasiticDisease/2617030')
ans1_3=task_1('RxInteractions/1966748')

print('task1')
print('Wounds/23778438',ans1_1[ans1_1.nonzero()])
print('ParasiticDisease/2617030',ans1_2[ans1_2.nonzero()])
print('RxInteractions/1966748',ans1_3[ans1_3.nonzero()])

def TF(tokens):
    d = {}
    for word in tokens:
        if not word in d:
            d[word] = 1
        else:
            d[word] += 1
    for word in d:
        d[word]=np.float(d[word])/len(tokens)
    return d
    
def IDF(RDD1,RDD2):
    # get the number of corpus from keyAndListOfWords
    N = RDD1.count()
    # RDD2 use allCounts
    idf=RDD2.map(lambda x: (x[0],np.log(np.float(N)/x[1])))
    return idf

idfs=IDF(keyAndListOfWords,allCounts)


def task_2(key):
    dictWord = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1] if x[0] == key))
    wordindict=dictWord.join(dictionary).map(lambda x: x[0]).collect()
    tf = TF(wordindict)
    def my_TFdict(letter):
        return tf[letter]
    tf_forIDF=sc.parallelize(list(tf))
    tf_forIDF=tf_forIDF.map(lambda x: (x,my_TFdict(x)))
    tfidf=tf_forIDF.join(idfs)
    tfidf = tfidf.map(lambda x: (x[0], x[1][0] * x[1][1]))
    ans=dictionary.join(tfidf).sortBy(lambda x:x[1][0])
    ans=np.array(ans.map(lambda x: x[1][1]).collect())
    return ans

ans2_1=task_2('PalliativeCare/16552238')
ans2_2=task_2('SquamousCellCancer/23991972')
ans2_3=task_2('HeartFailure/25940075')


WordToId = {}
IdToWord = {}
for i in dict:
    WordToId[i[0]] = i[1]
    IdToWord[i[1]] = i[0]

def TF_forall(tokens):
    d=[0.0]*20000
    for word in tokens:
        if word in WordToId:
            d[WordToId[word]]+=1
    N=sum(d)
    d=np.array(d)/N
    return list(d)

idf_dict=dictionary.leftOuterJoin(idfs).map(lambda x: x[1]).sortBy(lambda x: x[0]).map(lambda x: x[1])

def my_multiply(list1,list2):
    return [a*b for a,b in zip(list1,list2)]

def TFIDF_forall():
    tfs=keyAndListOfWords.map(lambda x: (x[0], TF_forall(x[1])))
    idf_dictl=idf_dict.collect()
    tfidf_all=tfs.map(lambda x: (x[0],my_multiply(idf_dictl,x[1])))
    return tfidf_all

def KNN(k,tfidf):
    test=tfidf.collect()
    tfidf_all=TFIDF_forall()
    distance=tfidf_all.map(lambda x:(x[0],np.linalg.norm(np.array(test)-np.array(x[1])))).sortBy(lambda x:x[1]).take(k)
    newlist=[]
    for i in range(k):
        newlist.append(distance[i][0].split('/')[0])
    def most_frequent(List): 
        occurence_count = Counter(List) 
        return occurence_count.most_common(1)[0][0] 
    return most_frequent(newlist)            

def predictLabel(k,s):
    Text = sc.parallelize(list([s]))
    Text = Text.map(lambda x: (regex.sub(' ', x).lower().split()))
    tfs = Text.map(lambda x: TF_forall(x))
    idf_dictl = idf_dict.collect()
    tfidf = tfs.map(lambda x: my_multiply(idf_dictl, x))
    return KNN(k, tfidf)

ans3_1=predictLabel (10, 'Simulation technology for health care professional skills training and assessment.  Changes in medical practice that limit instruction time and patient availability, the expanding options for diagnosis and management, and advances in technology are contributing to greater use of simulation technology in medical education. Four areas of high-technology simulations currently being used are laparoscopic techniques, which provide surgeons with an opportunity to enhance their motor skills without risk to patients; a cardiovascular disease simulator, which can be used to simulate cardiac conditions; multimedia computer systems, which includes patient-centered, case-based programs that constitute a generalist curriculum in cardiology; and anesthesia simulators, which have controlled responses that vary according to numerous possible scenarios. Some benefits of simulation technology include improvements in certain surgical technical skills, in cardiovascular examination skills, and in acquisition and retention of knowledge compared with traditional lectures. These systems help to address the problem of poor skills training and proficiency and may provide a method for physicians to become self-directed lifelong learners.')
ans3_2=predictLabel (10, 'Propofol inhibits T-helper cell type-2 differentiation by inducing apoptosis via activating gamma-aminobutyric acid receptor.  Propofol has been shown to attenuate airway hyperresponsiveness in asthma patients. Our previous study showed that it may alleviate lung inflammation in a mouse model of asthma. Given the critical role of T-helper cell type-2 (Th2) differentiation in asthma pathology and the immunomodulatory role of the gamma-aminobutyric acid type A (GABAFor in vivo testing, chicken ovalbumin-sensitized and challenged asthmatic mice were used to determine the effect of propofol on Th2-type asthma inflammation. For in vitro testing, Th2-type cytokines as well as the cell proliferation and apoptosis were measured to assess the effects of propofol on Th2 cell differentiation and determine the underlying mechanisms.We found that propofol significantly decreased inflammatory cell counts and interleukin-4 and inflammation score in vivo. Propofol, but not intralipid, significantly reduced the Th2-type cytokine interleukin-5 secretion and caused Th2 cell apoptosis without obvious inhibition of proliferation in vitro. A GABA receptor agonist simulated the effect of propofol, whereas pretreatment with an antagonist reversed this effect.This study demonstrates that the antiinflammatory effects of propofol on Th2-type asthma inflammation in mice are mediated by inducing apoptosis without compromising proliferation during Th2 cell differentiation via activation of the GABA receptor.Copyright Â© 2016 Elsevier Inc. All rights reserved.')
ans3_3=predictLabel (10, 'Evaluation of isopathic treatment of Salmonella enteritidis in poultry.  Salmonellosis is a common problem worldwide in commercially reared poultry. It is associated with human Salmonellosis. No fully satisfactory method of control is available.Nosodes to an antibiotic-resistant strain of Salmonella enterica serovar Enteritidis in D30 (30X) potency were prepared. One day old chicks (N = 180) were divided into four groups: two control and two different preparations of the nosode. Treatments were administered in drinking water for 10 days. The birds were challenged by a broth culture of the same Salmonella, by mouth, on day 17. Cloacal swabs were taken twice weekly for Salmonella enterica serovar Enteritidis.Birds receiving active treatment were less likely to grow the strain of Salmonella from cloacal swabs compared to control.Isopathy is low cost and non-toxic. It may have a role to play in the widespread problem of Salmonella in poultry. Further research should be conducted.')
ans3_4=predictLabel (10, 'Management of the neck after chemoradiotherapy for head and neck cancers in Asia: consensus statement from the Asian Oncology Summit 2009.  The addition of a planned neck dissection after radiotherapy has traditionally been considered standard of care for patients with positive neck-nodal disease. With the acceptance of chemoradiotherapy as the new primary treatment for patients with locally advanced squamous-cell head and neck cancers, and the increasing numbers of patients who achieve a complete response, the role of planned neck dissection is now being questioned. The accuracy and availability of a physical examination or of different imaging modalities to identify true complete responses adds controversy to this issue. This consensus statement will address some of the controversies surrounding the role of neck dissection following chemoradiotherapy for squamous-cell carcinomas of the head and neck, with particular reference to patients in Asia.')
ans3_5=predictLabel (10, '[Quality management in the hospital with special reference to the medical departments].  Present German health legislation requires standardisation of systems and procedures not only for medical departments but for entire clinics as well. Changes in organisation and treatment possibilities should enable clinic administrations to better manage the quality of the services provided. This system of quality control and quality management is based on the introduction of ISO 9004, part 2. Medical and non-medical procedures are standardised and streamlined through flow-diagrams, the details of which are summarised in a series of quality handbooks with an emphasis on guidelines and standards. These handbooks are distributed clinic-wide, and although acceptance is not yet determined, objections are probable. Future analysis will show the effects due to quality management in the clinical setting.')
ans3_6=predictLabel (10, 'Suicide attempts involving power drills.  A 61-year-old man was found dead next to a power drill soiled with blood and bone dust. A 5 mm circular wound of the forehead corresponded to the size of the drill bit. Subarachnoid haemorrhage was present over the anterior pole of the left frontal lobe with a penetrating injury extending 75 mm into the frontal lobe white matter towards, but not involving, the basal ganglia. No major intracranial vessels had been injured and there was no significant intraparenchymal haemorrhage. Death was due to haemorrhage from self-inflicted stab wounds to the abdomen with an associated penetrating intracranial wound from a power drill. Deaths due to power drills are rare and are either accidents or suicides. Wounds caused by power drills may be mistaken for bullet entrance wounds, and the marks around a wound from the drill chuck as muzzle imprints. A lack of internal bevelling helps to distinguish the entrance wound from that due to a projectile. Significant penetration of the brain may occur without lethal injury. Copyright © 2013 Elsevier Ltd and Faculty of Forensic and Legal Medicine. All rights reserved.')
ans3_7=predictLabel (10, 'Neurobehavioral recovery.  This review discusses recent programs in early and late neurobehavioral recovery from closed head injury (CHI). The research on early recovery has encompassed the relationship of localized brain lesions to the duration of impaired consciousness and features of posttraumatic amnesia. Of the research on late neurobehavioral outcome of CHI, studies emanating from the Traumatic Coma Data Bank are reviewed in detail, including analysis of acute neurologic indices in relation to recovery of memory, information processing speed, and other cognitive measures. Recent studies concerning the neurobehavioral outcome of CHI in children are discussed as are investigations of behavioral disturbance, psychosocial outcome, and family variables. The review concludes with an assessment of recent studies concerning the efficacy of rehabilitation directed toward the cognitive sequelae of CHI and preliminary trials to evaluate the potential use of psychoactive drugs in the postacute management of head injured patients.')
ans3_8=predictLabel (10, 'Ethical issues arising from the requirement to sign a consent form in palliative care.  French healthcare networks aim to help healthcare workers to take care of patients by improving cooperation, coordination and the continuity of care. When applied to palliative care in the home, they facilitate overall care including medical, social and psychological aspects. French legislation in 2002 required that an information document explaining the functioning of the network should be given to patients when they enter a healthcare network. The law requires that this document be signed. Ethical issues arise from this legislation with regard to the validity of the signature of dying patients. Signature of the consent form by a guardian or trustee, a designated person--the Person of Trust--transforms the doctor-patient relationship into a triangular doctor-patient-third-party relationship.')

print('task1')
print('Wounds/23778438',ans1_1[ans1_1.nonzero()])
print('ParasiticDisease/2617030',ans1_2[ans1_2.nonzero()])
print('RxInteractions/1966748',ans1_3[ans1_3.nonzero()])

print('task2')
print('PalliativeCare/16552238',ans2_1)
print('SquamousCellCancer/23991972',ans2_2)
print('HeartFailure/25940075',ans2_3)

print("task3")
print(ans3_1)
print(ans3_2)
print(ans3_3)
print(ans3_4)
print(ans3_5)
print(ans3_6)
print(ans3_7)
print(ans3_8)