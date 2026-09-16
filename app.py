"""Offline pipeline entry point (was 311 lines).

Run: ``python app.py`` — builds the LIF network from feather files and
writes ``fly_brain_results.png`` + ``fly_brain_spikes.npz``.
Tunables live in :class:`flybrain.config.OfflineConfig`.
"""
from flybrain.offline import main, run_offline

__all__ = ["main", "run_offline"]


if __name__ == "__main__":
    main()
