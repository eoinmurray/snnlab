"""The default snnviz house-style tokens."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    background: str = "#ffffff"
    ink: str = "#1a1a1a"
    accent: str = "#c8102e"
    muted: str = "#666666"
    rule: str = "#e7e5df"
    cyan: str = "#00b4d8"
    amber: str = "#e89400"
    dark_grey: str = "#3a3a3a"
    mid_grey: str = "#6a6a6a"

    def colour(self, role: str) -> str:
        try:
            return {
                "background": self.background,
                "ink": self.ink,
                "accent": self.accent,
                "muted": self.muted,
                "rule": self.rule,
                "cyan": self.cyan,
                "amber": self.amber,
                "dark_grey": self.dark_grey,
                "mid_grey": self.mid_grey,
            }[role]
        except KeyError as error:
            raise ValueError(f"unknown snnviz colour role: {role}") from error

    def apply(self) -> None:
        """Apply the house style used by new snnviz figures."""

        import matplotlib as mpl
        from cycler import cycler

        mpl.rcParams.update(
            {
                "font.family": "monospace",
                "font.monospace": [
                    "JetBrains Mono",
                    "IBM Plex Mono",
                    "Menlo",
                    "Consolas",
                    "Courier New",
                    "DejaVu Sans Mono",
                ],
                "font.size": 10.5,
                "axes.facecolor": self.background,
                "axes.edgecolor": self.ink,
                "axes.linewidth": 1.4,
                "axes.prop_cycle": cycler(
                    color=(
                        self.ink,
                        self.accent,
                        self.cyan,
                        self.amber,
                        self.dark_grey,
                        self.mid_grey,
                    )
                ),
                "figure.facecolor": self.background,
                "savefig.facecolor": self.background,
                "xtick.color": self.ink,
                "ytick.color": self.ink,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "legend.fancybox": False,
                "legend.framealpha": 1.0,
                "lines.solid_capstyle": "butt",
                "lines.solid_joinstyle": "miter",
            }
        )
