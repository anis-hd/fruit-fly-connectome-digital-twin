"""Live raster visualization for the offline pipeline (app.py)."""
import numpy as np


class LiveViz:
    def __init__(self, n_mon: int, window: int = 600, live: bool = True):
        self.ok, self.window = live, window
        self.img = np.zeros((window, n_mon))
        self.rate = np.zeros(window)
        if not live:
            return
        try:
            import matplotlib.pyplot as plt
            self._plt = plt
            plt.ion()
            self.fig, (self.a1, self.a2) = plt.subplots(
                2, 1, figsize=(12, 8), sharex=True)
            self.im = self.a1.imshow(self.img.T, aspect="auto", cmap="gray_r",
                                     vmin=0, vmax=1, interpolation="none")
            self.ln, = self.a2.plot(self.rate)
            self.a1.set_ylabel("neurons (sens|inter|motor)")
            self.a2.set_ylabel("pop. rate")
            self.a2.set_xlabel("time (ms)")
        except Exception:
            self.ok = False

    def update(self, t: int, mon_spikes, pop_rate: float):
        if not self.ok:
            return
        self.img = np.roll(self.img, -1, axis=0)
        self.img[-1] = mon_spikes
        self.rate = np.roll(self.rate, -1)
        self.rate[-1] = pop_rate
        try:
            self.im.set_data(self.img.T)
            self.ln.set_ydata(self.rate)
            self.a2.set_ylim(0, max(0.05, self.rate.max() * 1.2))
            self.a1.set_title(f"t = {t} ms")
            self.fig.canvas.draw_idle()
            self._plt.pause(0.001)
        except Exception:
            self.ok = False
