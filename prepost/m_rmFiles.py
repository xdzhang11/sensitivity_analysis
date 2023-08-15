# %% Remove result files
import os
import os.path
import shutil
nSeed = 12
n = int(1e4)
folderName = os.path.join('C:',os.sep,'Simulations', 'DTU10mw', 'mcs')
for j in range(0, 7501):
    for k in range(nSeed):
        selname = "case_%07d_%02d.sel"%(j,k)
        datname = "case_%07d_%02d.dat"%(j,k)
        htcname = "case_%07d_%02d.htc"%(j,k)
        filename1 = os.path.join(folderName,'res', selname)
        filename2 = os.path.join(folderName,'res', datname)
        filename3 = os.path.join(folderName,'htc', htcname)
        filename4 = os.path.join(folderName,'htc2', htcname)
        if not (os.path.exists(filename1) and os.path.exists(filename2)):
            print(htcname)
            shutil.copyfile(filename3, filename4)

#%%
import os
import os.path
import shutil
nSeed = 12
n = int(1e4)
folderName = os.path.join('C:',os.sep,'Simulations', 'DTU10mw', 'mcs')
for j in range(0, 7501):
    for k in range(nSeed):
        selname = "case_%07d_%02d.sel"%(j,k)
        datname = "case_%07d_%02d.dat"%(j,k)
        htcname = "case_%07d_%02d.htc"%(j,k)
        filename1 = os.path.join(folderName,'res', selname)
        filename2 = os.path.join(folderName,'res', datname)
        filename3 = os.path.join(folderName,'htc', htcname)
        filename4 = os.path.join(folderName,'htc2', htcname)
        if (os.path.exists(filename1) and os.path.getsize(filename1)<1858) or (os.path.exists(filename2) and os.path.getsize(filename2)<960000):
            print(htcname)
            shutil.copyfile(filename3, filename4)


#%%
import os
import os.path
import shutil
nSeed = 12
n = int(1e4)
folderName = os.path.join('C:',os.sep,'Simulations', 'DTU10mw', 'mcs')
for j in range(7501, n):
    for k in range(nSeed):
        selname = "case_%07d_%02d.sel"%(j,k)
        datname = "case_%07d_%02d.dat"%(j,k)
        filename1 = os.path.join('res', selname)
        filename2 = os.path.join('res', datname)
        if (os.path.exists(filename1) and os.path.getsize(filename1)<1858) or (os.path.exists(filename2) and os.path.getsize(filename2)<960000):
            os.remove(filename1)
            os.remove(filename2)



#%%



import os
import os.path
nSeed = 12
n = int(1e4)
for j in range(7501, n):
    for k in range(nSeed):
        selname = "case_%07d_%02d.sel"%(j,k)
        datname = "case_%07d_%02d.dat"%(j,k)
        htcname = "case_%07d_%02d.htc"%(j,k)
        filename1 = os.path.join('res', selname)
        filename2 = os.path.join('res', datname)
        filename3 = os.path.join('htc', htcname)
        if os.path.exists(filename1) and os.path.exists(filename2):
            try:
                os.remove(filename3)
            except OSError:
                pass
#%%

import os

#%%
dir_name = "htc"
items = os.listdir(dir_name)

for item in items:
    if item.endswith(".dat") or item.endswith(".sel"):
        os.remove(os.path.join(dir_name, item))



#%%
import os
import os.path
#%%
nSeed = 12
n = int(1e4)
for j in range(2500, 5001):
    for k in range(nSeed):
        selname = "case_%07d_%02d.sel"%(j,k)
        datname = "case_%07d_%02d.dat"%(j,k)
        filename1 = os.path.join('res', selname)
        filename2 = os.path.join('res', datname)
        try:
            os.remove(filename1)
        except OSError:
            pass
        try:
            os.remove(filename2)
        except OSError:
            pass
# %%
