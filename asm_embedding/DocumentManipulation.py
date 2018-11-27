import json
import re
import os

def list_to_str(li):
    i=''
    for x in li:
        i=i+' '+x
    i=i+' endfun'*5
    return i

def document_append(strin):
    with open('/Users/giuseppe/docuent_X86','a') as f:
        f.write(strin)

ciro=set()
cantina=[]
num_total=0
num_filtered=0
with open('/Users/giuseppe/dump.x86.linux.json') as f:
    l=f.readline()
    print('loaded')
    r = re.split('(\[.*?\])(?= *\[)', l)
    del l
    for x in r:
        if '[' in x:
            gennaro=json.loads(x)
            for materdomini in gennaro:
                num_total=num_total+1
                if materdomini[0] not in ciro:
                    ciro.add(materdomini[0])
                    num_filtered=num_filtered+1
                    a=list_to_str(materdomini[1])
                    document_append(a)
        del x
    print(num_total)
    print(num_filtered)