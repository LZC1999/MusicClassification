import matplotlib.pyplot as plt
import re


def draw(Loss, Acc):
    plt.title("Loss")
    plt.plot(range(len(Loss)), Loss, 'r', label="Loss")
    plt.show()
    plt.title("Acc")
    plt.plot(range(len(Acc)), Acc, 'g', label="Acc")
    plt.show()
