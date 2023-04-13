from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from tempfile import TemporaryDirectory
from thesispy.definitions import Collection

from thesispy.experiments.instance import MORunResult
from thesispy.experiments.validation import plot_deformed_mesh
from thesispy.elastix_wrapper.wrapper import execute_visualize, get_run_result_mo, populate_run_result

class PointCursor:
    def __init__(self, scatter, labels):
        self.scatter = scatter
        self.labels = labels
        self.ax = scatter.axes
        self.annotation = self.ax.annotate("", xy=(0,0), xytext=(20,20),
            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
        self.annotation.set_visible(False)

    def __call__(self, event):
        if not event.inaxes:
            return

        cont, ind = self.scatter.contains(event)
        if not cont:
            self.annotation.set_visible(False)
            self.ax.figure.canvas.draw_idle()
            return

        index = ind["ind"][0]
        x, y = self.scatter.get_offsets()[index]
        self.annotation.xy = (x, y)
        text = self.labels[index]
        self.annotation.set_text(text)
        self.annotation.set_visible(True)
        self.ax.figure.canvas.draw_idle()

class MORunResultVisualization:
    def __init__(self, moresult: MORunResult):
        self.moresult: MORunResult = moresult
        self.front = moresult.objective_values

    def onclick(self, event):
        x, y = event.xdata, event.ydata
        distances = (self.front - [x, y]) ** 2
        index = np.argmin(distances[:, 0] + distances[:, 1])
        self.plot_run_result(index)

    def plot(self):
        fig, ax = plt.subplots()
        scatter = ax.scatter([p[0] for p in self.front], [p[1] for p in self.front])
        labels = [f"({p[0]}, {p[1]})" for p in self.front]

        cursor = PointCursor(scatter, labels)
        fig.canvas.mpl_connect("motion_notify_event", cursor)
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        plt.show()

    def plot_run_result(self, index: int):
        with TemporaryDirectory() as out_dir:
            out_dir = Path(out_dir)
            run_result = self.moresult.approximation_set[index]
            populate_run_result(run_result, out_dir)
            execute_visualize(out_dir)
            slice_tuple = (slice(None), slice(None), run_result.instance.size[2] // 2)
            max_size = max(run_result.instance.size)
            fig, _ = plot_deformed_mesh(run_result, slice_tuple, max_size / 12)
            fig.show()


if __name__ == "__main__":
    mo_out_dir = Path("/home/joasiee/Documents/projects/elastix/experiments/output/1681408250_synthetic_1_mogomea_982275/out")
    run_result = get_run_result_mo(Collection.SYNTHETIC, 1, mo_out_dir)

    mo_result = MORunResultVisualization(run_result)
    mo_result.plot()
