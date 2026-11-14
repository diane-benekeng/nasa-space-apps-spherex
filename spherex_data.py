"""SPHEREx Level-2 data access and lightweight measurement helpers.

This module intentionally has no Streamlit dependency. A UI can import these
functions and decide how to cache, display, or persist the returned frames.
"""

from __future__ import annotations

from typing import TypedDict
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Row
from astropy.time import Time
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from pyvo.dal.exceptions import DALServiceError


COLLECTIONS = ("spherex_dr1", "spherex_qr3")
DEEP_COLLECTIONS = ("spherex_dr1_deep", "spherex_qr3_deep")
MAX_QUERY_ROWS = 256
MAX_DISPLAY_FRAMES = 48
GOOD_FLAG_VALUES = (0, 2, 2097152)

BARNARDS_STAR = {
    "ra": 269.4520769586187,
    "dec": 4.69336496657667,
    "pm_ra_cosdec": -801.551,
    "pm_dec": 10362.394,
    "parallax": 546.9759,
    "radial_velocity": -110.11,
    "reference_epoch": 2016.0,
}


class DataAccessError(RuntimeError):
    """Raised when IRSA or a SPHEREx FITS product cannot be read."""


class Frame(TypedDict):
    """One downloaded Level-2 image cutout and its measured metadata."""

    image: np.ndarray
    variance: np.ndarray
    flags: np.ndarray
    observation_time: str
    mjd: float
    band: str
    wavelength_um: float
    bandwidth_um: float
    flux: float | None
    uncertainty: float | None
    snr: float | None
    flagged_fraction: float
    target_x: float
    target_y: float


class BandScore(TypedDict):
    """Measured band ranking; flagged_fraction is a quality proxy."""

    band: str
    snr: float
    flagged_fraction: float
    score: float
    n_frames: int


class Spectrum(TypedDict):
    """Display-oriented spectrum; flux is aperture-mean MJy/sr."""

    wavelength_um: np.ndarray
    flux: np.ndarray
    uncertainty: np.ndarray


class Measurement(TypedDict):
    flux: float | None
    uncertainty: float | None
    snr: float | None
    flagged_fraction: float


class _Candidate(TypedDict):
    access_url: str
    mjd: float
    band: str


def resolve_target(name: str) -> SkyCoord:
    """Resolve a target name through Astropy's online name resolver."""
    return SkyCoord.from_name(name)


def _coordinate_at_mjd(
    coordinate: SkyCoord, mjd: float, track_barnards_star: bool
) -> SkyCoord:
    if not track_barnards_star:
        return coordinate

    reference = SkyCoord(BARNARDS_STAR["ra"], BARNARDS_STAR["dec"], unit="deg")
    if coordinate.separation(reference).arcsec > 1:
        raise ValueError(
            "Barnard's Star tracking requires its Gaia EDR3 reference coordinates."
        )

    barnards_star = SkyCoord(
        ra=coordinate.ra,
        dec=coordinate.dec,
        distance=(1000 / BARNARDS_STAR["parallax"]) * u.pc,
        pm_ra_cosdec=BARNARDS_STAR["pm_ra_cosdec"] * u.mas / u.yr,
        pm_dec=BARNARDS_STAR["pm_dec"] * u.mas / u.yr,
        radial_velocity=BARNARDS_STAR["radial_velocity"] * u.km / u.s,
        obstime=Time(BARNARDS_STAR["reference_epoch"], format="jyear"),
        frame="icrs",
    )
    return barnards_star.apply_space_motion(new_obstime=Time(mjd, format="mjd"))


def _column_value(row: Row, column: str) -> object:
    value = row[column]
    if np.ma.is_masked(value):
        raise ValueError(f"IRSA returned an empty {column} value.")
    return value


def _cutout_url(access_url: str, ra: float, dec: float, fov: float) -> str:
    parts = urlsplit(access_url)
    parameters = dict(parse_qsl(parts.query, keep_blank_values=True))
    parameters.update({"center": f"{ra:.10f},{dec:.10f}", "size": f"{fov:.6f}"})
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(parameters), parts.fragment)
    )


def _bracket(axis: np.ndarray, coordinate: float) -> tuple[int, float]:
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError("The WCS-WAVE control-point axis is invalid.")
    if coordinate < axis[0] or coordinate > axis[-1]:
        raise ValueError("The target falls outside the WCS-WAVE control-point grid.")
    lower = int(np.searchsorted(axis, coordinate, side="right") - 1)
    lower = min(max(lower, 0), axis.size - 2)
    fraction = (coordinate - axis[lower]) / (axis[lower + 1] - axis[lower])
    return lower, float(fraction)


def _interpolate_wavelength(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Bilinearly interpolate WCS-WAVE wavelength and bandwidth in micrometres."""
    x_index, x_fraction = _bracket(x_axis, x)
    y_index, y_fraction = _bracket(y_axis, y)
    if values.shape != (y_axis.size, x_axis.size, 2):
        raise ValueError("The WCS-WAVE values do not match their control-point grid.")

    corners = values[
        y_index : y_index + 2,
        x_index : x_index + 2,
        :,
    ].astype(float)
    if not np.all(np.isfinite(corners)):
        raise ValueError("The WCS-WAVE grid contains non-finite values at the target.")

    lower = corners[0, 0] * (1 - x_fraction) + corners[0, 1] * x_fraction
    upper = corners[1, 0] * (1 - x_fraction) + corners[1, 1] * x_fraction
    result = lower * (1 - y_fraction) + upper * y_fraction
    return float(result[0]), float(result[1])


def _wavelength_at_position(
    hdus: fits.HDUList, coordinate: SkyCoord
) -> tuple[float, float, float, float]:
    image_hdu = hdus["IMAGE"]
    x_pixel, y_pixel = WCS(image_hdu.header).world_to_pixel(coordinate)
    x_pixel, y_pixel = float(x_pixel), float(y_pixel)
    height, width = image_hdu.data.shape
    if not (-0.5 <= x_pixel < width - 0.5 and -0.5 <= y_pixel < height - 0.5):
        raise ValueError("The requested coordinate is outside the returned cutout.")

    header = image_hdu.header
    x_detector = 1 + x_pixel - float(header["CRPIX1A"])
    y_detector = 1 + y_pixel - float(header["CRPIX2A"])
    wave_row = hdus["WCS-WAVE"].data[0]
    wavelength, bandwidth = _interpolate_wavelength(
        np.asarray(wave_row["X"], dtype=float),
        np.asarray(wave_row["Y"], dtype=float),
        np.asarray(wave_row["VALUES"], dtype=float),
        x_detector,
        y_detector,
    )
    return wavelength, bandwidth, x_pixel, y_pixel


def measure_frame(
    image: np.ndarray,
    variance: np.ndarray,
    flags: np.ndarray,
    x: float | None = None,
    y: float | None = None,
) -> Measurement:
    """Estimate aperture-mean brightness, uncertainty, S/N, and flagged fraction."""
    if image.ndim != 2 or image.shape != variance.shape or image.shape != flags.shape:
        raise ValueError("IMAGE, VARIANCE, and FLAGS must be matching 2-D arrays.")

    height, width = image.shape
    x = (width - 1) / 2 if x is None else x
    y = (height - 1) / 2 if y is None else y
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    aperture = radius <= 2
    edge = min(x, y, width - 1 - x, height - 1 - y)
    outer_radius = min(20.0, edge)
    inner_radius = min(8.0, outer_radius / 2)
    background_ring = (radius >= inner_radius) & (radius <= outer_radius)
    if outer_radius <= 4 or not np.any(aperture):
        return {
            "flux": None,
            "uncertainty": None,
            "snr": None,
            "flagged_fraction": 1.0,
        }

    aperture_count = int(np.count_nonzero(aperture))
    good_flags = np.isin(flags, GOOD_FLAG_VALUES)
    flagged_fraction = float(
        np.count_nonzero(aperture & ~good_flags) / aperture_count
    )
    valid = (
        good_flags
        & np.isfinite(image)
        & np.isfinite(variance)
        & (variance >= 0)
    )
    good_aperture = aperture & valid
    good_background = background_ring & valid
    n_aperture = int(np.count_nonzero(good_aperture))
    if n_aperture == 0 or np.count_nonzero(good_background) < 10:
        return {
            "flux": None,
            "uncertainty": None,
            "snr": None,
            "flagged_fraction": flagged_fraction,
        }

    background = image[good_background]
    background_level = float(np.median(background))
    flux = float(np.mean(image[good_aperture]) - background_level)
    variance_error = float(np.sqrt(np.sum(variance[good_aperture])) / n_aperture)
    background_error = float(np.std(background, ddof=1) / np.sqrt(n_aperture))
    uncertainty = max(variance_error, background_error)
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return {
            "flux": None,
            "uncertainty": None,
            "snr": None,
            "flagged_fraction": flagged_fraction,
        }

    return {
        "flux": flux,
        "uncertainty": uncertainty,
        "snr": float(flux / uncertainty),
        "flagged_fraction": flagged_fraction,
    }


def get_frames(
    ra: float,
    dec: float,
    fov: float,
    max_frames: int = 12,
    include_deep: bool = False,
    track_barnards_star: bool = False,
) -> list[Frame]:
    """Search IRSA SIA2 and download a time-spread set of Level-2 cutouts."""
    if not 0 <= ra <= 360:
        raise ValueError("Right ascension must be between 0 and 360 degrees.")
    if not -90 <= dec <= 90:
        raise ValueError("Declination must be between -90 and 90 degrees.")
    if not 0.05 <= fov <= 0.5:
        raise ValueError("Field of view must be between 0.05 and 0.5 degrees.")
    if track_barnards_star and fov < 0.1:
        raise ValueError(
            "Barnard's Star tracking requires a field of view of at least 0.1 degrees."
        )
    if not 1 <= max_frames <= MAX_DISPLAY_FRAMES:
        raise ValueError(f"Frame limit must be between 1 and {MAX_DISPLAY_FRAMES}.")

    coordinate = SkyCoord(ra % 360, dec, unit="deg")
    search_radius = fov * np.sqrt(2) / 2
    collections = COLLECTIONS + (DEEP_COLLECTIONS if include_deep else ())
    candidates: dict[str, _Candidate] = {}
    for collection in collections:
        try:
            results = Irsa.query_sia(
                pos=(coordinate, search_radius * u.deg),
                collection=collection,
                calib_level=2,
                maxrec=MAX_QUERY_ROWS,
            )
        except (DALServiceError, OSError, TimeoutError, URLError) as error:
            raise DataAccessError(
                f"IRSA query failed for {collection}: {error}"
            ) from error

        for row in results:
            access_url = str(_column_value(row, "access_url"))
            if access_url in candidates:
                continue
            candidates[access_url] = {
                "access_url": access_url,
                "mjd": float(_column_value(row, "t_min")),
                "band": str(_column_value(row, "energy_bandpassname")),
            }

    ordered = sorted(
        candidates.values(),
        key=lambda item: (item["mjd"], item["access_url"]),
    )
    if len(ordered) > max_frames:
        indices = np.linspace(0, len(ordered) - 1, max_frames, dtype=int)
        ordered = [ordered[index] for index in indices]

    frames: list[Frame] = []
    for candidate in ordered:
        cutout_url = _cutout_url(
            candidate["access_url"], ra % 360, dec, fov
        )
        mjd = candidate["mjd"]
        target_coordinate = _coordinate_at_mjd(
            coordinate, mjd, track_barnards_star
        )
        try:
            file_path = download_file(
                cutout_url,
                cache=True,
                show_progress=False,
                timeout=90,
            )
            with fits.open(file_path, memmap=False) as hdus:
                image = np.asarray(hdus["IMAGE"].data, dtype=np.float32).copy()
                variance = np.asarray(hdus["VARIANCE"].data, dtype=np.float32).copy()
                flags = np.asarray(hdus["FLAGS"].data).copy()
                wavelength, bandwidth, x_pixel, y_pixel = _wavelength_at_position(
                    hdus, target_coordinate
                )
                measurement = measure_frame(
                    image, variance, flags, x_pixel, y_pixel
                )
        except (OSError, TimeoutError, URLError, KeyError, ValueError) as error:
            raise DataAccessError(
                f"Could not read the SPHEREx cutout {candidate['access_url']}: {error}"
            ) from error

        frames.append(
            {
                "image": image,
                "variance": variance,
                "flags": flags,
                "observation_time": Time(mjd, format="mjd").isot,
                "mjd": mjd,
                "band": candidate["band"],
                "wavelength_um": wavelength,
                "bandwidth_um": bandwidth,
                **measurement,
                "target_x": x_pixel,
                "target_y": y_pixel,
            }
        )
    return frames


def get_all_bands() -> list[str]:
    """Return SPHEREx's six detector bandpass labels."""
    return [f"SPHEREx-D{detector}" for detector in range(1, 7)]


def rank_bandpasses(frames: list[Frame]) -> list[BandScore]:
    """Rank by median aperture S/N, downweighted by flagged-pixel fraction."""
    grouped: dict[str, list[Frame]] = {}
    for frame in frames:
        if frame["snr"] is not None:
            grouped.setdefault(frame["band"], []).append(frame)

    scores = []
    for band, band_frames in grouped.items():
        snr = float(
            np.median(
                [frame["snr"] for frame in band_frames if frame["snr"] is not None]
            )
        )
        flagged_fraction = float(
            np.median([frame["flagged_fraction"] for frame in band_frames])
        )
        scores.append(
            {
                "band": band,
                "snr": snr,
                "flagged_fraction": flagged_fraction,
                "score": max(snr, 0.0) * (1.0 - flagged_fraction),
                "n_frames": len(band_frames),
            }
        )
    return sorted(
        scores,
        key=lambda score: (-score["score"], -score["snr"], score["band"]),
    )


def explain_choice(best_band: BandScore) -> str:
    """Explain the measured ranking without treating flags as contaminants."""
    if best_band["score"] <= 0:
        return (
            "No band has a positive median aperture S/N after flag weighting; "
            "no best band is selected."
        )
    return (
        f"{best_band['band']} ranks first (median aperture S/N "
        f"{best_band['snr']:.1f}, flagged-aperture fraction "
        f"{best_band['flagged_fraction']:.0%}); the flag fraction is a conservative "
        "quality proxy, not a confirmed contaminant count."
    )


def get_spectrum(
    ra: float,
    dec: float,
    fov: float = 0.1,
    max_frames: int = 48,
    include_deep: bool = False,
    track_barnards_star: bool = False,
) -> Spectrum:
    """Assemble a display spectrum using each cutout's WCS-WAVE mapping."""
    frames = get_frames(
        ra, dec, fov, max_frames, include_deep, track_barnards_star
    )
    samples = [
        frame
        for frame in frames
        if frame["flux"] is not None and frame["uncertainty"] is not None
    ]
    samples.sort(key=lambda frame: frame["wavelength_um"])
    return {
        "wavelength_um": np.asarray(
            [frame["wavelength_um"] for frame in samples], dtype=float
        ),
        "flux": np.asarray([frame["flux"] for frame in samples], dtype=float),
        "uncertainty": np.asarray(
            [frame["uncertainty"] for frame in samples], dtype=float
        ),
    }
