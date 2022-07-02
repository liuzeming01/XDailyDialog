import random
import os
dir = "./data"
max_len = 5000
indices = random.sample(list(range(max_len)), 1000)
print("sample lenght: {}".format(len(indices)))

for file in os.listdir(dir):
    if "dialogues_" not in file:
        continue
    file = os.path.join(dir, file)
    print(file)
    data = open(file, 'r', encoding="utf-8").readlines()
    print("{} number: {}".format(file, len(data)))
    assert max_len < len(data)
    new_data = []
    for indice in indices:
        new_data.append(data[indice])
    print("new data length: {}".format(len(new_data)))
    f = open(file, "w", encoding="utf-8")
    f.writelines(new_data)
    f.close()
