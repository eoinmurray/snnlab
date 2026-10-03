"""Named, deterministic figure layouts for stills and animation frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .styles import Theme


@dataclass(frozen=True)
class FigureRect:
    """A rectangle in normalized figure coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def mpl(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height

    def inset(
        self,
        padding: float | tuple[float, float, float, float],
    ) -> "FigureRect":
        """Inset by relative left, bottom, right and top fractions."""

        if isinstance(padding, (int, float)):
            left = bottom = right = top = float(padding)
        else:
            left, bottom, right, top = padding
        if min(left, bottom, right, top) < 0:
            raise ValueError("figure-grid padding must be non-negative")
        width = self.width * (1 - left - right)
        height = self.height * (1 - bottom - top)
        if width <= 0 or height <= 0:
            raise ValueError("figure-grid padding consumes the region")
        return FigureRect(
            self.x + self.width * left,
            self.y + self.height * bottom,
            width,
            height,
        )


@dataclass(frozen=True)
class FigureRegion:
    """A named rectangular selection of grid tracks."""

    name: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1
    reserved: bool = False


def _tracks(value: int | Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"figure-grid {label} must be positive")
        return (1.0,) * value
    tracks = tuple(float(item) for item in value)
    if not tracks or any(item <= 0 for item in tracks):
        raise ValueError(f"figure-grid {label} must contain positive weights")
    return tracks


@dataclass
class FigureGrid:
    """Place named regions on weighted rows and columns.

    Rows are numbered from top to bottom; columns from left to right. Gaps and
    bounds use normalized figure coordinates. Regions must be rectangular and
    may span any number of tracks.
    """

    rows: int | Sequence[float]
    columns: int | Sequence[float]
    bounds: FigureRect | tuple[float, float, float, float] = FigureRect(
        0.06, 0.06, 0.88, 0.88
    )
    row_gap: float | Sequence[float] = 0.03
    column_gap: float | Sequence[float] = 0.03
    theme: Theme = field(default_factory=Theme)
    _regions: dict[str, FigureRegion] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.rows = _tracks(self.rows, "rows")
        self.columns = _tracks(self.columns, "columns")
        self.row_gap = self._gaps(self.row_gap, len(self.rows) - 1, "row")
        self.column_gap = self._gaps(
            self.column_gap, len(self.columns) - 1, "column"
        )
        if not isinstance(self.bounds, FigureRect):
            self.bounds = FigureRect(*self.bounds)
        if self.bounds.width <= 0 or self.bounds.height <= 0:
            raise ValueError("figure-grid bounds must have positive size")
        if sum(self.column_gap) >= self.bounds.width:
            raise ValueError("figure-grid column gaps consume the bounds")
        if sum(self.row_gap) >= self.bounds.height:
            raise ValueError("figure-grid row gaps consume the bounds")

    @staticmethod
    def _gaps(
        value: float | Sequence[float], count: int, label: str
    ) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            gaps = (float(value),) * count
        else:
            gaps = tuple(float(item) for item in value)
        if len(gaps) != count or any(item < 0 for item in gaps):
            raise ValueError(
                f"figure-grid {label} gaps must contain {count} non-negative values"
            )
        return gaps

    def place(
        self,
        name: str,
        *,
        row: int,
        column: int,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> FigureRegion:
        """Place one named region, rejecting overlaps and invalid spans."""

        return self._place(
            FigureRegion(name, row, column, rowspan, colspan, reserved=False)
        )

    def reserve(
        self,
        name: str,
        *,
        row: int,
        column: int,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> FigureRegion:
        """Reserve tracks that must not become plotting axes."""

        return self._place(
            FigureRegion(name, row, column, rowspan, colspan, reserved=True)
        )

    def _place(self, region: FigureRegion) -> FigureRegion:
        if not region.name or region.name in self._regions:
            raise ValueError("figure-grid region names must be non-empty and unique")
        if region.row < 0 or region.column < 0:
            raise ValueError("figure-grid row and column must be non-negative")
        if region.rowspan <= 0 or region.colspan <= 0:
            raise ValueError("figure-grid spans must be positive")
        if region.row + region.rowspan > len(self.rows):
            raise ValueError(f"figure-grid region {region.name!r} exceeds its rows")
        if region.column + region.colspan > len(self.columns):
            raise ValueError(f"figure-grid region {region.name!r} exceeds its columns")
        cells = self._cells(region)
        for existing in self._regions.values():
            if cells & self._cells(existing):
                raise ValueError(
                    f"figure-grid region {region.name!r} overlaps {existing.name!r}"
                )
        self._regions[region.name] = region
        return region

    @staticmethod
    def _cells(region: FigureRegion) -> set[tuple[int, int]]:
        return {
            (row, column)
            for row in range(region.row, region.row + region.rowspan)
            for column in range(region.column, region.column + region.colspan)
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._regions)

    def region(self, name: str) -> FigureRegion:
        try:
            return self._regions[name]
        except KeyError as error:
            raise KeyError(f"unknown figure-grid region: {name}") from error

    def rect(
        self,
        name: str,
        *,
        padding: float | tuple[float, float, float, float] = 0.0,
    ) -> FigureRect:
        """Resolve a named region to normalized figure coordinates."""

        region = self.region(name)
        widths = self._sizes(self.columns, self.bounds.width, self.column_gap)
        heights = self._sizes(self.rows, self.bounds.height, self.row_gap)
        x = self.bounds.x + sum(widths[: region.column])
        x += sum(self.column_gap[: region.column])
        top = self.bounds.y + self.bounds.height
        top -= sum(heights[: region.row]) + sum(self.row_gap[: region.row])
        width = sum(widths[region.column : region.column + region.colspan])
        width += sum(
            self.column_gap[
                region.column : region.column + region.colspan - 1
            ]
        )
        height = sum(heights[region.row : region.row + region.rowspan])
        height += sum(self.row_gap[region.row : region.row + region.rowspan - 1])
        return FigureRect(x, top - height, width, height).inset(padding)

    @staticmethod
    def _sizes(
        tracks: Sequence[float], extent: float, gaps: Sequence[float]
    ) -> tuple[float, ...]:
        available = extent - sum(gaps)
        scale = available / sum(tracks)
        return tuple(track * scale for track in tracks)

    def subgrid(
        self,
        name: str,
        *,
        rows: int | Sequence[float],
        columns: int | Sequence[float],
        padding: float | tuple[float, float, float, float] = 0.0,
        row_gap: float | Sequence[float] = 0.02,
        column_gap: float | Sequence[float] = 0.02,
    ) -> "FigureGrid":
        """Create a nested grid inside an existing non-reserved region."""

        region = self.region(name)
        if region.reserved:
            raise ValueError("cannot create a subgrid inside a reserved region")
        return FigureGrid(
            rows,
            columns,
            bounds=self.rect(name, padding=padding),
            row_gap=row_gap,
            column_gap=column_gap,
            theme=self.theme,
        )

    def figure(
        self,
        *,
        figsize: tuple[float, float],
        dpi: int = 120,
    ) -> Any:
        """Create an opaque figure using the default snnviz house style."""

        self.theme.apply()
        figure = plt.figure(figsize=figsize, dpi=dpi)
        figure.patch.set_facecolor(self.theme.background)
        return figure

    def add_axes(
        self,
        figure: Any,
        name: str,
        *,
        padding: float | tuple[float, float, float, float] = 0.0,
        frame: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Create a Matplotlib axis in one named region."""

        region = self.region(name)
        if region.reserved:
            raise ValueError(f"cannot create axes in reserved region {name!r}")
        axis = figure.add_axes(self.rect(name, padding=padding).mpl, **kwargs)
        self.style_axis(axis, frame=frame)
        return axis

    def style_axis(self, axis: Any, *, frame: bool = True) -> None:
        axis.set_facecolor(self.theme.background)
        for spine in axis.spines.values():
            spine.set_visible(frame)
            spine.set_color(self.theme.ink)
            spine.set_linewidth(1.4)
        axis.tick_params(
            colors=self.theme.ink,
            direction="in",
            width=1.0,
            length=4,
        )

    def draw_region(
        self,
        figure: Any,
        name: str,
        *,
        role: str = "ink",
        fill: str | None = None,
        linewidth: float = 1.4,
        dashed: bool = False,
        zorder: float = -10,
    ) -> Rectangle:
        """Draw the hard-edged house-style boundary of a named region."""

        colour = self.theme.colour(role)
        patch = Rectangle(
            self.rect(name).mpl[:2],
            self.rect(name).width,
            self.rect(name).height,
            transform=figure.transFigure,
            facecolor=fill or self.theme.background,
            edgecolor=colour,
            linewidth=linewidth,
            linestyle=(0, (5, 4)) if dashed else "solid",
            zorder=zorder,
            clip_on=False,
        )
        figure.add_artist(patch)
        return patch
