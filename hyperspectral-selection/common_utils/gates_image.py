import os
import json
import uuid

import numpy as np
from matplotlib import pyplot as plt


class GatesImageCreator:
    def __init__(self, dir_path, dataset_name, n_bands):
        self.dir_path = dir_path
        self.gates_number_to_idx = {}
        self.n_bands = n_bands
        self.dataset_name = dataset_name

    def gates_progression_image(self,matrix):
        #plt.clf()
        plt.imshow(matrix, cmap='gray_r', vmin=0, vmax=1.0)
        #plt.axis('off')  # Turn off axis labels and ticks
        plt.xlabel('bands idx')
        plt.ylabel('n bands')
        ymax = max(self.gates_number_to_idx.keys())
        ymin = min(self.gates_number_to_idx.keys())
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax = plt.gca()

        # Hide X and Y axes label marks
        #ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        #ax.set_xticks([])
        ax.set_yticks([])

        ax.set_ylim([ymax, ymin])
        #plt.axis([ymin, ymax])
        plt.savefig(f'/home/dsi/yanivz/hyperspectral-selection/{self.dataset_name}/grayscale_image_final_gates_guid_{str(uuid.uuid4())}.png')


    def create_gates_matrix(self):
        gates_matrix = np.zeros((self.n_bands,))
        for k,v in reversed(sorted(self.gates_number_to_idx.items())):
            for gates in v:
                if int(k) != sum(np.array(gates) > 0):
                    continue
                gates_matrix = np.vstack((gates_matrix, np.array(gates)))
        return gates_matrix

    def append_data_to_gates_number_to_idx(self,data):
        for k_t,v in data.items():
            k=int(k_t)
            if k in self.gates_number_to_idx.keys():
                self.gates_number_to_idx[k].append(v)
            else:
                self.gates_number_to_idx[k]=[v]

    def iterate_files(self):
        file_names = os.listdir(self.dir_path)
        for name in file_names:
            path = os.path.join(self.dir_path,name)
            with open(path,mode='r') as f:
                data = json.load(f)
                self.append_data_to_gates_number_to_idx(data)

if __name__=='__main__':
    gCreator = GatesImageCreator('/home/dsi/yanivz/hyperspectral-selection/Salinas_7/gates',dataset_name='Salinas_7',n_bands=204)
    gCreator.iterate_files()
    print(gCreator.gates_number_to_idx)
    matrix = gCreator.create_gates_matrix()
    print(matrix)
    gCreator.gates_progression_image(matrix)
