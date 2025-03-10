import os
from typing import Dict
import pandas as pd
class ResultsSaver:
    def __init__(self,dataset_name,optimizer_name):
        self.dataset_name = dataset_name
        self.optimizer_name = optimizer_name

    def split_per_class(self,di,avg_key,std_key,new_key):
        if avg_key in di.keys():
            for i in range(di[avg_key].shape[0]):
                di[f'{new_key}_{i}']=f'{di[avg_key][i]:.2f}+-{di[std_key][i]:.2f}'
            del di[avg_key]
            del di[std_key]
    def split_f1(self,di):
        f1_key = 'F1 scores avg'
        f1_key_std = 'F1 scores std'
        new_key = "f1_class"
        self.split_per_class(di=di,avg_key=f1_key,std_key=f1_key_std,new_key=new_key)
    def split_acc(self,di):
        acc_key = 'Acc per class avg'
        acc_key_std = 'Acc per class std'
        new_key = "acc_class"
        self.split_per_class(di=di,avg_key=acc_key,std_key=acc_key_std,new_key=new_key)


    def save(self, di:Dict,method_name:str):
        file_name = f'{self.dataset_name}_{self.optimizer_name}.csv'
        header = True

        self.split_f1(di)
        self.split_acc(di)
        df = pd.DataFrame({method_name:di}).transpose()
        if os.path.exists(file_name):
            header=False
        df.to_csv(file_name, mode='a', header=header)

        print(di)
