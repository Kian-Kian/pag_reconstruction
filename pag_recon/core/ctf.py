"""CTF parsing and paired-map validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = (
    "Phase",
    "X",
    "Y",
    "Bands",
    "Error",
    "Euler1",
    "Euler2",
    "Euler3",
    "MAD",
    "BC",
    "BS",
)


@dataclass(frozen=True)
class CTFMap:
    path: Path
    header: dict[str, str]
    phase_definitions: tuple[str, ...]
    columns: tuple[str, ...]
    values: np.ndarray

    @property
    def width(self) -> int:
        return int(self.header["XCells"])

    @property
    def height(self) -> int:
        return int(self.header["YCells"])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def x_step(self) -> float:
        return float(self.header["XStep"])

    @property
    def y_step(self) -> float:
        return float(self.header["YStep"])

    @property
    def project(self) -> str:
        return self.header.get("Prj", "")

    def column(self, name: str, *, dtype: np.dtype | type | None = None) -> np.ndarray:
        try:
            index = self.columns.index(name)
        except ValueError as exc:
            raise KeyError(f"Column {name!r} is absent from {self.path}") from exc
        result = self.values[:, index].reshape(self.shape)
        return result.astype(dtype, copy=False) if dtype is not None else result

    @property
    def euler_degrees(self) -> np.ndarray:
        indices = [self.columns.index(name) for name in ("Euler1", "Euler2", "Euler3")]
        return self.values[:, indices].reshape((*self.shape, 3)).astype(np.float32, copy=False)


def _split_header_line(line: str) -> tuple[str, str]:
    key, separator, value = line.partition("\t")
    return key.strip(), value.strip() if separator else ""


def read_ctf(path: str | Path) -> CTFMap:
    """Read an Oxford/AZtec channel text file while preserving its metadata."""

    path = Path(path)
    header_lines: list[str] = []
    column_line_index: int | None = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.rstrip("\r\n")
            header_lines.append(line)
            if line.startswith("Phase\tX\tY\t"):
                column_line_index = line_index
                break

    if column_line_index is None:
        raise ValueError(f"Could not find the CTF column header in {path}")

    header: dict[str, str] = {}
    phase_count: int | None = None
    phase_definitions: list[str] = []
    for line_index, line in enumerate(header_lines[:column_line_index]):
        key, value = _split_header_line(line)
        if key:
            header[key] = value
        if key == "Phases":
            phase_count = int(value)
            phase_definitions = header_lines[line_index + 1: line_index + 1 + phase_count]

    for required in ("XCells", "YCells", "XStep", "YStep"):
        if required not in header:
            raise ValueError(f"Missing {required!r} in CTF header: {path}")

    columns = tuple(header_lines[column_line_index].split("\t"))
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Missing required CTF columns {missing} in {path}")

    values = np.loadtxt(
        path,
        delimiter="\t",
        skiprows=column_line_index + 1,
        dtype=np.float32,
        ndmin=2,
        encoding="utf-8-sig",
    )
    expected_rows = int(header["XCells"]) * int(header["YCells"])
    if values.shape != (expected_rows, len(columns)):
        raise ValueError(
            f"Expected {(expected_rows, len(columns))} numeric values in {path}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite numeric values detected in {path}")

    return CTFMap(
        path=path,
        header=header,
        phase_definitions=tuple(phase_definitions),
        columns=columns,
        values=values,
    )


def paired_fields_match(alpha: CTFMap, gamma: CTFMap) -> dict[str, int | bool]:
    if alpha.shape != gamma.shape:
        return {"shape_match": False}
    result: dict[str, int | bool] = {"shape_match": True}
    for name in ("X", "Y", "MAD", "BC", "BS"):
        alpha_values = alpha.column(name)
        gamma_values = gamma.column(name)
        difference = np.abs(alpha_values - gamma_values)
        result[f"{name.lower()}_mismatch"] = int(
            np.count_nonzero(~np.isclose(alpha_values, gamma_values,
                             rtol=0.0, atol=1e-6, equal_nan=True))
        )
        result[f"{name.lower()}_max_abs_difference"] = float(np.nanmax(difference))
    return result


def _header_and_data_lines(path: Path) -> tuple[list[str], list[str], int]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Phase\tX\tY\t"):
            return lines[: index + 1], lines[index + 1:], index
    raise ValueError(f"Could not find the CTF column header in {path}")


def _replace_phase_definitions(
    source_header: list[str], template: CTFMap
) -> list[str]:
    phase_line = next(
        (index for index, line in enumerate(source_header) if line.startswith("Phases\t")),
        None,
    )
    if phase_line is None:
        raise ValueError("Source CTF header has no Phases entry")
    source_count = int(source_header[phase_line].split("\t", 1)[1])
    return [
        *source_header[:phase_line],
        f"Phases\t{len(template.phase_definitions)}",
        *template.phase_definitions,
        *source_header[phase_line + 1 + source_count:],
    ]


def infer_phase_id(template: CTFMap, phase_name: str = "Austenite") -> int:
    """Return the 1-based phase ID containing ``phase_name`` in a CTF template."""

    matches = [
        index
        for index, definition in enumerate(template.phase_definitions, start=1)
        if phase_name.casefold() in definition.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {phase_name!r} phase in {template.path}, got {matches}"
        )
    return matches[0]


def write_reconstructed_ctf(
    source_alpha: str | Path | CTFMap,
    gamma_template: str | Path | CTFMap,
    output_path: str | Path,
    euler_degrees: np.ndarray,
    reconstructed_mask: np.ndarray,
    *,
    gamma_phase_id: int | None = None,
) -> None:
    """Write a reconstructed gamma CTF while preserving alpha measurement fields.

    Only Phase and the three Euler columns are replaced in data rows. Source
    coordinates and quality columns remain byte-for-byte unchanged. Pixels
    outside ``reconstructed_mask`` are emitted as unindexed phase zero.
    """

    alpha = read_ctf(source_alpha) if not isinstance(source_alpha, CTFMap) else source_alpha
    template = (
        read_ctf(gamma_template)
        if not isinstance(gamma_template, CTFMap)
        else gamma_template
    )
    euler = np.asarray(euler_degrees, dtype=np.float32)
    mask = np.asarray(reconstructed_mask, dtype=bool)
    if euler.shape != (*alpha.shape, 3):
        raise ValueError(f"euler_degrees must have shape {(*alpha.shape, 3)}")
    if mask.shape != alpha.shape:
        raise ValueError(f"reconstructed_mask must have shape {alpha.shape}")
    if not np.isfinite(euler[mask]).all():
        raise ValueError("reconstructed Euler angles must be finite")
    phase_id = infer_phase_id(template) if gamma_phase_id is None else int(gamma_phase_id)
    if not 1 <= phase_id <= len(template.phase_definitions):
        raise ValueError("gamma_phase_id is outside the template phase definitions")

    source_header, data_lines, _ = _header_and_data_lines(alpha.path)
    if len(data_lines) != alpha.width * alpha.height:
        raise ValueError("Source CTF numeric row count does not match its grid")
    output_header = _replace_phase_definitions(source_header, template)
    column_index = {name: alpha.columns.index(name) for name in REQUIRED_COLUMNS}
    flat_euler = euler.reshape(-1, 3)
    flat_mask = mask.ravel()
    output_rows: list[str] = []
    for row_index, line in enumerate(data_lines):
        fields = line.split("\t")
        if len(fields) != len(alpha.columns):
            raise ValueError(f"Malformed numeric row {row_index + 1} in {alpha.path}")
        if flat_mask[row_index]:
            fields[column_index["Phase"]] = str(phase_id)
            for angle_index, name in enumerate(("Euler1", "Euler2", "Euler3")):
                fields[column_index[name]] = f"{flat_euler[row_index, angle_index]:.6f}"
        else:
            fields[column_index["Phase"]] = "0"
            for name in ("Euler1", "Euler2", "Euler3"):
                fields[column_index[name]] = "0.000000"
        output_rows.append("\t".join(fields))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join((*output_header, *output_rows)) + "\n", encoding="utf-8-sig"
    )
