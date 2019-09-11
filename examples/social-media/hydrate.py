import os
import sys
import time
from math import ceil
import json

from twython import Twython

APP_KEY = os.getenv('APP_KEY')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

fname = sys.argv[1]

twitter = Twython(APP_KEY, ACCESS_TOKEN)

def drawProgressBar(percent, barLen = 20):
    # draw a progress bar
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

with open(fname, 'r') as file:
    ids = [line.strip().split(',')[1] for line in file]

id_ints = []
for id in ids[1:]:
    try:
        id_ints.append(int(id.strip('"')))
    except ValueError as e:
        pass


BATCH = 100
hl = len(id_ints)
cycles = ceil(hl / BATCH)
with open(sys.argv[2], 'w') as wf:
    for i in range(0, cycles): ## iterate through all tweets
        statuses = [] #initialize data object
        h = id_ints[0:BATCH]
        del id_ints[0:BATCH]
        incremental = twitter.lookup_status(id=h) # each call gets 100 tweets
        statuses.extend(incremental)
        drawProgressBar(i / cycles, 40)

        for s in statuses:
            wf.write(json.dumps(s) + '\n')

        time.sleep(4) # 4 seconds for app auth  (300/15min.)
