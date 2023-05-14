import os
import shutil

def write_lines(data, file):
    print("write {} data into: {}".format(len(data), file))
    f = open(file, "w", encoding="utf8")
    f.writelines(data)
    f.close()

def split(root):
    print("split file in :{}".format(root))
    for file in os.listdir(root):
        if "train" not in file:
            continue
        train = open(os.path.join(root, file)).readlines()
        dev_start = int(len(train) * 0.7)
        test_start = int(len(train) * 0.85)
        dev = train[dev_start:test_start]
        test = train[test_start:]
        train = train[:dev_start]
        write_lines(train, os.path.join(root, "train."+file.split(".")[1]))
        write_lines(train, os.path.join(root, "dev."+file.split(".")[1]))
        write_lines(train, os.path.join(root, "test."+file.split(".")[1]))

if __name__ == '__main__':
    split("./data/multilingual")
    for lan in os.listdir("./data/monolingual"):
        split(os.path.join("./data/monolingual", lan))
    for lan in os.listdir("./data/crosslingual"):
        split(os.path.join("./data/crosslingual", lan))
    os.mkdir("./data/raw")
    shutil.move("./data/monolingual", "./data/raw/monolingual")
    shutil.move("./data/crosslingual", "./data/raw/crosslingual")
    shutil.move("./data/multilingual", "./data/raw/multilingual")
