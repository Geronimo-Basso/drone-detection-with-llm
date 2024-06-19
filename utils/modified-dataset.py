import os

path = '../originales-400-modifiled-txt'
files = os.listdir(path)

for file in files:
    if file.endswith(".txt"):
        complete_path = os.path.join(path,file)
        f = open(complete_path, 'r')
        read = f.readlines()
        if read:
            del read[0]

        print(read)

