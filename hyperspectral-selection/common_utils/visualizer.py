import numpy as np
import matplotlib.pyplot as plt
import os.path
from PIL import Image
import uuid
import json
from sklearn.metrics import auc

class Visualizer:
    def gates_progression_image(self, dataset, matrix, fold, lam, one_gates):
        plt.clf()
        plt.imshow(matrix, cmap='gray_r', vmin=0, vmax=1)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.savefig(f'/home/dsi/yanivz/hyperspectral-selection/{dataset}/iter_gates_progression_image_gates{one_gates}_fold{fold}_lam{lam}_guid_{str(uuid.uuid4())}.png')

    def gates_progression_image_pil(self, dataset, matrix, fold, lam, one_gates):
        # Scale the matrix values to the range [0, 255]
        scaled_matrix = ((1-matrix) * 255).astype(np.uint8)

        # Create a Pillow image from the matrix
        image = Image.fromarray(scaled_matrix, mode='L')

        # Save or display the image
        image.save(f'{dataset}/grayscale_image_gates_{one_gates}_fold_{fold}_lam_{lam}_guid_{str(uuid.uuid4())}.png')

    def write_last_gates(self, dataset, gates, one_gates):
        dir = f'{dataset}/gates'
        data = {int(one_gates): gates.tolist()}
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(f'{dir}/final_gates_{one_gates}_guid_{str(uuid.uuid4())}.json', "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def save_acc_plot(self, dataset, train, val, algo_name, fold, lam, optimizer):
        x_values = np.arange(len(train))
        plt.clf()
        plt.plot(x_values, train, label=f'Train_{fold}')
        plt.plot(x_values, val, label=f'Test_{fold}')
        title=f'acc_plot_{algo_name}_lam_{lam}_fold_{fold}_opt_{optimizer.__class__.__name__}'
        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(title)
        # Add a legend
        plt.legend()
        full_name=f'{dataset}/{title}_{str(uuid.uuid4())}'
        # Save the plot
        plt.savefig(f'{full_name}.png')
        with open(f'{full_name}.json', "w") as f:
            json.dump({"train_acc":train,"val_acc":val}, f, ensure_ascii=False)

    def auc(self,bs_acc, bs_n_bands):
        bs_to_auc={}
        for i, name in enumerate(bs_acc.keys()):
            if len(bs_n_bands[name]) == 1:
                bs_to_auc[name] = bs_acc[name][0]/100
            else:
                bs_to_auc[name] = auc(np.array(bs_n_bands[name]), np.array(bs_acc[name]))/ (100 * (bs_n_bands[name][-1] - bs_n_bands[name][0]))
        return bs_to_auc


    def draw_bs_methods_acc(self,bs_acc, bs_n_bands,path):
        bs_auc = self.auc(bs_acc,bs_n_bands)
        for i,name in enumerate(bs_acc.keys()):
            print(name)
            # Acc plot
            color = plt.plot(bs_n_bands[name], bs_acc[name],
                             label=f'{name} AUC: {bs_auc[name]:.4f}')[0].get_color()
            # Auc plot
            #plt.text(0.95, 0.95 - 0.05 * i,
            #         f'{name} AUC: {bs_auc[name]/(bs_n_bands[name][-1]-bs_n_bands[name][0]):.2f}%',
            #         transform=plt.gca().transAxes,color=color)

        plt.suptitle('Accuracy over Different BS Methods')
        plt.xlabel('Bands')
        plt.ylabel('Accuracy')
        plt.legend()
        print("Hello")
        #plt.show()
        plt.savefig(path)

