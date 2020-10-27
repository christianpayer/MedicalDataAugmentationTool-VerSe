
import utils.io.landmark

import numpy as np
import pickle


if __name__ == '__main__':
    #     0-6: C1-C7
    #     7-18: T1-12
    #     19-24: L1-6
    #     25: T13
    num_landmarks = 26
    landmark_order = list(range(19)) + [25] + list(range(19, 25))
    possible_successors = [set() for _ in range(num_landmarks)]
    landmarks = utils.io.landmark.load_csv('../verse2020_dataset/setup/landmarks.csv', num_landmarks, 3)
    for key, value in landmarks.items():
        for i in range(num_landmarks):
            if not value[i].is_valid:
                continue
            for j in landmark_order[landmark_order.index(i)+1:]:
                if not value[j].is_valid:
                    continue
                else:
                    possible_successors[i].add(j)
                    break
    possible_successors[5].add(7)
    print(possible_successors)

    offsets = [{} for _ in range(num_landmarks)]
    c_t_offsets = []
    t_l_offsets = []
    for key, value in landmarks.items():
        min_coord = np.array([10000.0, 10000.0, 10000.0])
        max_coord = np.array([0.0, 0.0, 0.0])
        for i in range(num_landmarks):
            if not value[i].is_valid:
                continue
            for j in landmark_order[landmark_order.index(i)+1:]:
                if not value[j].is_valid:
                    continue
                else:
                    if j not in offsets[i]:
                        offsets[i][j] = []
                    curr_offset = value[i].coords - value[j].coords
                    offsets[i][j].append(curr_offset)
                    if (i == 5 and j == 7) or (i == 6 and j == 7):
                        c_t_offsets.append(curr_offset)
                    if (i == 17 and j == 19) or (i == 18 and j == 19) or (i == 25 and j == 19):
                        t_l_offsets.append(curr_offset)
                    min_coord = np.min([min_coord, value[j].coords], axis=0)
                    max_coord = np.max([max_coord, value[j].coords], axis=0)
                    break

    offsets[5][7] = c_t_offsets
    offsets[6][7] = c_t_offsets
    offsets[17][19] = t_l_offsets
    offsets[18][19] = t_l_offsets
    offsets[25][19] = t_l_offsets
    mean_offsets = [{} for _ in range(num_landmarks)]
    means_distances = [{} for _ in range(num_landmarks)]
    std_distances = [{} for _ in range(num_landmarks)]
    for i, landmark_offset in enumerate(offsets):
        for j, values in landmark_offset.items():
            mean = np.mean(values, axis=0)
            distance = np.linalg.norm(mean)
            distances = [np.linalg.norm(value) for value in values]
            mean_offsets[i][j] = mean / distance
            means_distances[i][j] = np.mean(distances)
            std_distances[i][j] = np.std(distances)

    with open('possible_successors.pickle', 'wb') as f:
        pickle.dump(possible_successors, f)

    with open('units_distances.pickle', 'wb') as f:
        pickle.dump((mean_offsets, means_distances, std_distances), f)
    print('done')
