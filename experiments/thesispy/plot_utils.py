import matplotlib
matplotlib.use("module://mplcairo.qt")

import matplotlib.pyplot as plt
from thesispy.definitions import ROOT_DIR

import seaborn as sns
sns.set_palette(sns.color_palette("colorblind"))
plt.style.use(ROOT_DIR / "resources/plt_custom.txt")

def mesh_size_as_str(mesh_size):
    return f"{mesh_size[0]+3}x{mesh_size[1]+3}x{mesh_size[2]+3}"
