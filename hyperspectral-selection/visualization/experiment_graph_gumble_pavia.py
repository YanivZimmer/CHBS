from common_utils.visualizer import Visualizer
bs_acc = {
   'gumble': [97.36671962	,98.26278039,	99.34287272,	99.2640171,	98.71441042,	99.12781524,	99.76582778,	99.42412553],
'SPA-BS':[	93.07989675,	94.91028577	,98.66902389,	98.9509984,	99.0680818,	99.43604241,	99.68936222,	99.37871016],
'BS-NETS':	[95.49330728,	95.83264317	,98.8410729	,99.46474366,	99.20666972	,99.44801554,	99.68458639	,99.39064845],
'TAttMSRecNet':	[95.52917156,	96.46590442,	99.17799845,	99.74910052,	99.84229305,	99.87574357,	99.73237497,	99.86379386],
'DARecNet-BS':	[96.53994224,	97.17796333,	99.1612749,	99.65590855,	99.2544623,	99.66306845, 99.55790735, 99.79211013],
'BS-Net-Conv':	[96.15763243,	97.28786514,	98.03342169,	99.21622395,	99.17561125,	99.50058167,	99.70369971	,99.43607011],
'DRL':	[94.48255174,	97.91632116,	98.53283317,	98.56149872,	98.89842485,	98.94380225,	99.60095123	,99.49581297],
'PCA':	[94.76927953,	95.15401908,	95.56502042,	96.67137626,	98.71442755,	98.07162891,	98.64513217	,98.56386679],
'SpaBS':	[94.20295872,	94.75256254,	97.57938132,	99.13498599,	98.54000848,	99.45995269,	98.99640064	,99.15647994],
'SNMF':[	94.27468671	,97.8255341,	99.10630387	,99.38589632	,99.89247312	,99.82795214	,99.70609005	,99.77538515]
}
bs_n_bands = {
    'gumble': [3,4,5,6,7,8,9,10],
'SPA-BS': [3,4,5,6,7,8,9,10],
'BS-NETS': [3,4,5,6,7,8,9,10],
'TAttMSRecNet': [3,4,5,6,7,8,9,10],
'DARecNet-BS': [3,4,5,6,7,8,9,10],
'BS-Net-Conv': [3,4,5,6,7,8,9,10],
'DRL': [3,4,5,6,7,8,9,10],
'PCA': [3,4,5,6,7,8,9,10],
'SpaBS': [3,4,5,6,7,8,9,10],
'SNMF': [3,4,5,6,7,8,9,10],
}
if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.draw_bs_methods_acc(bs_acc=bs_only_EGHBS,bs_n_bands=bs_n_bands_EGHBS,path='/home/dsi/yanivz/hyperspectral-selection/visualization/pavia5_bs_gumble.png')