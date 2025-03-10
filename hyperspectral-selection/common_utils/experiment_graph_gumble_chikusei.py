from visualizer import Visualizer
bs_acc = {
   'gumble': [98.94188973,	99.58114347	,99.25893301,99.78090367,	99.79507943,	99.66491848,	99.66104635,	99.73837831],
'SPA-BS':[96.49578209,	98.72925464,	98.99603113,	99.70100502,	99.8479230,	99.5553603,	99.74610753,	99.87885253],
'BS-NETS':	[96.93652026,	99.28472821,	99.621092,	99.29244323,	99.69970996,	99.69068926,	99.75,	99.93556219],
}
bs_n_bands = {
    'gumble': [3,4,5,6,7,8,9,10],
'SPA-BS': [3,4,5,6,7,8,9,10],
'BS-NETS': [3,4,5,6,7,8,9,10],
}
if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.draw_bs_methods_acc(bs_acc=bs_acc,bs_n_bands=bs_n_bands,path='/home/dsi/yanivz/hyperspectral-selection/visualization/chik5_bs_gumble.png')