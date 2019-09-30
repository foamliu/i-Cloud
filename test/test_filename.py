import os
import random

filename = 'abc.jpg'

basename = os.path.basename(filename)
print(basename)

tokens = filename.split('.')
name = tokens[0]
ext = tokens[1]
print(name, ext)

rand = random.randint(10000, 99999)
filename = '{}_{}.{}'.format(name, rand, ext)
print(filename)
