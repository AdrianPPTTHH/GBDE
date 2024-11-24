import numpy as np
import pandas as pd
import random
from tqdm import tqdm


def get_replace_word_dict(word_balls):
    replace_word_dict = {}

    for word_ball in tqdm(word_balls, desc="Generate Candidate..."):
        for word_emb in word_ball:
            if( int(word_emb[1]) not in replace_word_dict ):
                replace_word_dict[int(word_emb[1])] = [int(word_emb_temp[1]) for word_emb_temp in word_ball if word_emb_temp[1] != word_emb[1]]
            else:
                replace_word_dict[int(word_emb[1])].extend([int(word_emb_temp[1]) for word_emb_temp in word_ball if word_emb_temp[1] != word_emb[1]])

    for word in tqdm(replace_word_dict.keys(), desc="Removing duplicates..."):
        replace_word_dict[word] = list(set(replace_word_dict[word]))

    return replace_word_dict

class BallDict:
    def __init__(self, centers, balls, path, distance=0.1):
        self.centers = centers
        self.balls = balls
        # self.percentage = percentage
        self.distance = distance
        self.distance_matrix = np.load(path)
        self.replace_dict = {}
    
    def get_distance(self, x1, x2):
        
        a = int(x1[1])
        b = int(x2[1])

        return self.distance_matrix[a][b]
    
    def get_center_distance(self, x1, center):
        data1_without_index = x1[2:]
        data2_without_index = center[2:]
        return np.linalg.norm(data1_without_index - data2_without_index)  
    
    def sort_balls_by_center(self):
        sorted_word_balls = []
        for i, array in enumerate(self.balls):
            center = self.centers[i] 
            sorted_array = sorted(array, key=lambda x: self.distance_to_center(x, center))
            sorted_word_balls.append(sorted_array)
        self.balls = sorted_word_balls
    
    def sort_replace_dict(self):
        for key, values in tqdm(self.replace_dict.items(), desc="Sorting replace_dict"):
            self.replace_dict[key] = sorted(values, key=lambda x: self.distance_matrix[key][x])

    def set_replace_dict(self):    
        for word in tqdm(self.replace_dict.keys(), desc="Unrepeated..."):
            # self.replace_dict[word] = list(set(self.replace_dict[word]))[:50]
            self.replace_dict[word] = list(set(self.replace_dict[word]))


    def get_replace_word_dict(self):

        for i, word_ball in tqdm(enumerate(self.balls), desc="Generate candidate word list...", total=len(self.balls)):

            for word_emb in word_ball:
                # ensure that the projection of the center replacement word to center original word vector is a positive number (angle<90 °)
                # self.replace_dict[int(word_emb[1])] = []
                # vec1 = (self.centers[i] - word_emb)[2:]

                # for word_emb_temp in word_ball:
                #     if word_emb_temp[1] != word_emb[1]:
                #         vec2 = (self.centers[i] - word_emb_temp)[2:]

                #         unit_vec1 = vec1 / np.linalg.norm(vec1)
                #         projection = np.dot(vec2, unit_vec1) * unit_vec1
                #         # angle = np.arccos(np.dot(projection, unit_vec1))

                #         angle = np.arccos(-np.dot(projection, vec1) / (np.linalg.norm(projection) * np.linalg.norm(vec1)))
                #         angle_degrees = np.degrees(angle)
                        
                #         if np.allclose(angle_degrees, 0.0, atol=1e-2):
                #             self.replace_dict[int(word_emb[1])].append(int(word_emb_temp[1]))

                if(int(word_emb[1]) not in self.replace_dict):
                    self.replace_dict[int(word_emb[1])] = [int(word_emb_temp[1]) for word_emb_temp in word_ball ]
                else:
                    self.replace_dict[int(word_emb[1])].extend([int(word_emb_temp[1]) for word_emb_temp in word_ball ])

        self.set_replace_dict()

        self.sort_replace_dict()

        np.save('./Attacker/replace.npy', self.replace_dict)

        return self.replace_dict
    

if __name__ == "__main__":

    embedding_path = '../counter-fitted-vectors.txt' 

    embeddings = []
    with open(embedding_path, 'r') as ifile:
        for line in ifile:
            embedding = [float(num) for num in line.strip().split()[1:]]
            embeddings.append(embedding)
    embeddings = np.array(embeddings, dtype="float32")

    # 计算欧式距离矩阵
    n = embeddings.shape[0]
    euclidean_distances = np.zeros((n, n), dtype="float32")
    for i in range(n):
        for j in range(n):
            euclidean_distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

    np.save('./euclidean_distances.npy', euclidean_distances)

    centers = np.load("./cluster_center_result.npy", allow_pickle=True)
    word_balls = np.load("./cluster_result.npy", allow_pickle=True)
    Ba = BallDict(centers=centers[0], balls=word_balls, path="./euclidean_distances.npy", distance=0.5)
    replace_dict = Ba.get_replace_word_dict()
