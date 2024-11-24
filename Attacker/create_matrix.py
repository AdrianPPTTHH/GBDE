import numpy as np
from tqdm import tqdm
from granular_ball.HyperBallCluster import main
# from granular_ball.HyperBallCluster import calculate_distances
import datetime
import pickle

start_time = datetime.datetime.now()

idx2word = {}
word2idx = {}

path = "../counter-fitted-vectors.txt"
# path = './glove.6B.200d.txt'

embedding_matrix = []

with open(path, 'r') as file:
    for line in file:
        elements = line.split()
        vector = np.array(elements[1:], dtype=float)
        embedding_matrix.append(vector)

embedding_matrix = np.array(embedding_matrix)

np.save('embedding.npy', embedding_matrix)

print("Building vocab...")
with open(path, 'r') as ifile:
    for line in ifile:
        word = line.split()[0]
        if word not in idx2word:
            idx2word[len(idx2word)] = word
            word2idx[word] = len(idx2word) - 1
                
#Load embedding
data = np.load('./embedding.npy')

data = data[:-1]

x = np.array(range(0,data.shape[0])).reshape(-1,1)

data_with_sequnce = np.concatenate((x,data),axis=1)

np.save("./data_with_label.npy", data_with_sequnce)

data = np.load("./data_with_label.npy")
print("data.shape:",data.shape)

split_num = 1 #Cut the data into split_num blocks (If direct clustering requires a particularly large amount of video memory)

min_synonyms = 3 #Ensure that there are no empty particles

datas = np.array_split(data, split_num)
result = []
centers = []
last_label = 0

for data_epoch in datas:
    result_temp, centers_temp = main(data_epoch, min_synonyms = min_synonyms)
    last_label += len(data_epoch)

    if(last_label != 0):
        for arr in tqdm(result_temp, total=len(result_temp), desc="Splicing in progress...."):
            for row in arr:
                row[0] += last_label

    result.extend(result_temp)
    centers.append(centers_temp)

with open('cluster_center_result.npy', 'wb') as f1:
    pickle.dump(centers, f1)

with open('cluster_result.npy', 'wb') as f:
    pickle.dump(result, f)


end_time = datetime.datetime.now()

duration = end_time - start_time

print(f"Running time：{duration}")