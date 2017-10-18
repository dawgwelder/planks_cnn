import os

with open('train.txt', 'r') as tr:
    s = tr.readlines()
tmp = ''
for i, item in enumerate(s):
    tmp +=item[1:]

with open('wtrain.txt', 'w') as f:
    f.write(tmp)

with open('test.txt', 'r') as tr:
    s = tr.readlines()
tmp = ''
for i, item in enumerate(s):
    tmp +=item[1:]

with open('wtest.txt', 'w') as f:
    f.write(tmp)