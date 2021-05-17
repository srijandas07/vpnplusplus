import pandas as pd
import csv
import os
import math

f = open('/data/stars/user/sdas/smarthomes_data/splits/test_new_CS.txt', "r")
dir1 = '/data/stars/user/rdai/smarthomes/Blurred_smarthome_clipped_SSD/'
outfile = open('test_Labels.csv', "w")
outf = csv.writer(outfile, delimiter=',')
outf.writerow(['name', 'start', 'end'])
for i in f.readlines():
    n_frames = len(os.listdir(dir1 + os.path.splitext(i.strip())[0]))
    div = int(math.ceil(n_frames//128))
    #print(n_frames, div)
    if div>0:
        for j in range(0, div):
            t = 128
            outf.writerow([os.path.splitext(i.strip())[0], j*t, (j+1)*t])
    else:
        outf.writerow([os.path.splitext(i.strip())[0], '0', int(n_frames)])


