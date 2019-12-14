import h5py
filename = "data/data_prepro.h5"
f = h5py.File(filename,'r')
print(f.keys())
datasetnames =  [n for n in f.keys()]
for n in datasetnames:
    print(n)

#for item in f.keys():
#    print(item+":"+f[item])
