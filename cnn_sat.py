import numpy as np
np.random.seed(2)



def main():
    make_train_and_val_listfiles()


def load_labels():
    f = open('label_atx.txt','r')
    output = {}
    for line in f:
        tmp = line.split(',')
        filename = tmp[0].strip()
        truth = tmp[1].rstrip()
        if truth=='`': truth=0
        else: truth=1
        output[filename] = truth
    return output


def make_train_and_val_listfiles(frac_val=0.2):
    from glob import glob
    label_dict = load_labels()
    labels = label_dict.values()
    frac_pos = 1.*np.sum(labels) / len(labels)
    print frac_pos
    filenames = label_dict.keys()
    np.random.shuffle(filenames)
    train_labels = []
    f_train = open('data/train_listfile','w')
    f_val = open('data/val_listfile','w')
    n_pos_val = 0
    n_pos_train = 0
    for filename in filenames:
        # format is subfolder1/file1.JPEG 7
        this_label = label_dict[filename]
        this_name = 'atx/%s'%filename

        if (np.random.random()<frac_val):
            # validation
            f_val.write('%s %i\n'%(this_name, this_label))
            n_pos_val += this_label
        else:
            # training
            # balance the training set.
            # i.e. if this is a negative instance, throw it out with some probability.
            if ((this_label==0) & (np.random.random() > frac_pos)): continue
            f_train.write('%s %i\n'%(this_name, this_label))
            train_labels.append(this_label)
            n_pos_train += this_label
    f_train.close()
    f_val.close()
    print '%i different classes in training set'%(len(set(train_labels)))
    print '%i positive cases in training set'%(n_pos_train)
    print '%i positive cases in validation set'%(n_pos_val)


def zip_train_val():
   x = load_labels()
   filenames = x.keys()
   command = 'zip data_train_val '
   for filename in filenames:
       command += 'data/atx/%s '%(filename)
       command += ' '
   from os import system
   system(command)

if __name__ == '__main__':
    main()
