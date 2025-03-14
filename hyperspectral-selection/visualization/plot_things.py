from matplotlib import pyplot as plt


def dict_to_list(di):
    li = []
    for k, v in di.items():
        li.append(v)
    return li


if __name__ == "__main__":
    algo_kappa = {
        "ISSC": [0.9084, 0.9097, 0.9267, 0.9229, 0.9257, 0.9318],
        "MMCA": [0.8388, 0.8763, 0.9081, 0.9172, 0.9134, 0.9295],
        "LP": [0.8266, 0.9179, 0.9329, 0.9237, 0.9315, 0.9345],
        "WALUMI": [0.916, 0.9194, 0.9278, 0.9068, 0.9196, 0.9374],
        "WALUDI": [0.9082, 0.9266, 0.9374, 0.9352, 0.9342, 0.9402],
        "STG": [0.8711, 0.9186, 0.8612, 0.9144, 0.9157, 0.9218],
    }
    algo_acc = {
        "ISSC": [92.9820, 93.0194, 94.3753, 94.0714, 94.3052, 94.7634],
        "MMCA": [87.7548, 90.5694, 92.9352, 93.6179, 93.2953, 94.5810],
        "LP": [85.7075, 93.7067, 94.8382, 94.1368, 94.7353, 94.9597],
        "WALUMI": [93.6132, 93.8236, 94.4548, 92.8885, 93.8423, 95.2122],
        "WALUDI": [92.9773, 94.3566, 95.1888, 95.01122, 94.9410, 95.3993],
        "STG": [90.0131, 93.7394, 89.3678, 93.3934, 93.5431, 93.9872],
    }

    for algo_name, acc in algo_acc.items():
        plt.plot([10, 15, 20, 25, 30, 35], acc, label=algo_name)
    plt.legend()
    plt.savefig("benchmark-algorithms results.png")
    plt.show()
    # acc=test_bands_mlp(range(1,103))
    # print(acc)
