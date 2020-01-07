import json
from matplotlib import pyplot as plt


if __name__ == '__main__':
    f = "final_model_control_history.json"
    file = open(f, "r")
    vals = json.load(file)
    print(vals["dice_coef"])
    loss = vals["loss"].values()
    values = vals["dice_coef"].values()
    plt.plot(values)
    plt.xlabel("Numer epoki")
    plt.ylabel("Współczynnik Sorensena-Dice'a")
    plt.show()
