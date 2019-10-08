import datetime
import os
import random

if __name__ == '__main__':
    rand = random.randint(1000, 9999)
    print(rand)
    mydir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mydir = '{}_{}'.format(mydir, rand)
    print(mydir)

