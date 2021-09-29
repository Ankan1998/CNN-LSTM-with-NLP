import pickle

def Objectpickler(fname,object):
    with open(fname,'wb') as f:
        pickle.dump(object,f)

def pickleloader(fname):
    with open(fname, 'rb') as f:
        object = pickle.load(f)
    return object

if __name__=="__main__":
    # mnl = pickleloader(r'C:\Users\Ankan\Desktop\Github\RCNN\saved\list.pkl')
    # print(mnl)
    pass