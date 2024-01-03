import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_matrix(file_path):
    matrix = np.loadtxt(file_path)
    return matrix


def update(frame):
    plt.clf()
    plt.axis('off')
    plt.imshow(
        matrices[frame],
        cmap="plasma",
        interpolation="nearest",
    )


file_path = "b.txt"
matrices = []
current_matrix = []
with open(file_path, "r") as file:
    for line in file:
        if line.strip():  # Ignora le righe vuote
            row = np.fromstring(line.strip(), sep=" ")
            current_matrix.append(row)
        else:
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []

fig, ax = plt.subplots()
animation = FuncAnimation(fig, update, frames=len(matrices), interval=200)

plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.show()
