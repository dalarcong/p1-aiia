"""
Frame capture from public webcams (task 1.2), including stream URL
resolution from SkylineWebcams pages.

Shared interface for every part of the assignment. Returns the frame
as JPEG bytes, ready to be saved to disk, sent to the VM or uploaded
to Blob Storage.
"""
import os
import re
import urllib.request
from urllib.parse import urlparse
# Silence harmless H.264 warnings
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
import cv2


OPEN_TIMEOUT_MS = 10000
READ_TIMEOUT_MS = 5000

PAGE_TIMEOUT_S = 10
SKYLINE_STREAM_HOST = "https://hd-auth.skylinewebcams.com/"
SKYLINE_SOURCE_RE = re.compile(r"source:'(livee\.m3u8\?a=[^']+)'")


class CaptureError(Exception):
    """Raised when a frame cannot be captured from the stream."""


def capture_frame(url: str, quality: int = 90) -> bytes:
    """Capture a single frame from a video stream and return it as JPEG.

    The stream is opened, read once and released on every call, so
    OpenCV's internal buffer never returns stale frames between captures.

    Args:
        url: Direct stream URL (.m3u8, MJPEG...), not the web page URL.
        quality: JPEG quality from 0 to 100.

    Returns:
        The frame encoded as JPEG bytes.

    Raises:
        CaptureError: If the stream cannot be opened, or the frame cannot
            be read or encoded.
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
    ])

    try:
        if not cap.isOpened():
            raise CaptureError("Could not open the stream")

        ret, frame = cap.read()
        if not ret:
            raise CaptureError("Could not read the frame")

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        if not ok:
            raise CaptureError("Could not encode the image")

        return buffer.tobytes()

    finally:
        cap.release()


def resolve_stream_url(url: str) -> str:
    """Resolve a SkylineWebcams page URL into its direct HLS stream URL.
    
    SkylineWebcams pages do not expose a stable stream URL: the page HTML
    contains a temporary token, and the stream URL is built from it. This
    function downloads the page, extracts the token andd returns the full
    stream URL.

    URLs from other sites are returned unchanged, so direct stream URLs
    can also be used.

    Args:
        url: SkylineWebcams page URL or direct stream URL.

    Returns:
        The direct stream URL.
    
    Raises:
        CaptureError: If the page cannot be downloaded or the stream
            source is not found in it.
    """
    if urlparse(url).netloc != "www.skylinewebcams.com":
        return url

    try:
        with urllib.request.urlopen(url, timeout=PAGE_TIMEOUT_S) as response:
            html = response.read().decode("utf-8", errors="replace")
    except OSError as e:
        raise CaptureError(f"Could not download the camera page: {e}") from e

    match = SKYLINE_SOURCE_RE.search(html)
    if match is None:
        raise CaptureError("Stream source not found in the camera page")

    return SKYLINE_STREAM_HOST + match.group(1).replace("livee.", "live.", 1)

