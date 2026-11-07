"""
Random access to the byte-split iSign pose ZIP hosted on Hugging Face.

The 4 HF part files are byte-splits of ONE zip. Instead of downloading ~170 GB,
we read the zip's central directory once and then fetch each member with a
single HTTP range request.
"""
import struct
import sys
import time
import zipfile
import zlib

import requests
from huggingface_hub import hf_hub_url


class RemoteMultipartZip:
    """File-like view over the joined part files (used by zipfile to read the index)."""

    READAHEAD = 256 * 1024  # zipfile does many tiny header reads; serve them from one request

    def __init__(self, repo_id, part_filenames, token, part_sizes=None):
        self.token = token
        self.session = requests.Session()
        self.parts, offset = [], 0
        for i, fname in enumerate(part_filenames):
            url = hf_hub_url(repo_id=repo_id, filename=fname, repo_type="dataset")
            size = part_sizes[i] if part_sizes else self._probe_size(url)
            self.parts.append({"url": url, "start": offset, "end": offset + size, "size": size})
            offset += size
        self.total_size = offset
        self._pos = 0
        self._buf_start, self._buf = 0, b""

    # ---- HTTP -------------------------------------------------------------- #
    def _get(self, url, byte_range, retries=5):
        headers = {"Authorization": f"Bearer {self.token}", "Range": byte_range}
        for attempt in range(retries):
            try:
                r = self.session.get(url, headers=headers, timeout=60)
                if r.status_code in (401, 403):
                    sys.exit(f"HTTP {r.status_code}: token invalid or dataset license not accepted.")
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _probe_size(self, url):
        return int(self._get(url, "bytes=0-0").headers["Content-Range"].split("/")[-1])

    def sizes(self):
        return [p["size"] for p in self.parts]

    def fetch(self, start, end):
        """Bytes [start, end) of the joined archive, crossing part boundaries if needed."""
        out, pos = bytearray(), start
        while pos < end:
            part = next(p for p in self.parts if p["start"] <= pos < p["end"])
            lo = pos - part["start"]
            hi = min(end, part["end"]) - part["start"] - 1
            out.extend(self._get(part["url"], f"bytes={lo}-{hi}").content)
            pos = part["start"] + hi + 1
        return bytes(out)

    # ---- file protocol used by zipfile ------------------------------------- #
    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self._pos = [offset, self._pos + offset, self.total_size + offset][whence]
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total_size - self._pos
        end = min(self._pos + n, self.total_size)
        if end <= self._pos:
            return b""
        if not (self._buf_start <= self._pos and end <= self._buf_start + len(self._buf)):
            fetch_end = min(max(end, self._pos + self.READAHEAD), self.total_size)
            self._buf_start, self._buf = self._pos, self.fetch(self._pos, fetch_end)
        data = self._buf[self._pos - self._buf_start : end - self._buf_start]
        self._pos = end
        return data


def read_member(reader, info: zipfile.ZipInfo) -> bytes:
    """Read one zip member with a single range request and verify its CRC."""
    name_len = len(info.orig_filename.encode("utf-8"))
    start = info.header_offset
    blob = reader.fetch(start, min(start + 30 + name_len + 1024 + info.compress_size, reader.total_size))
    if blob[:4] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    fn_len, extra_len = struct.unpack("<HH", blob[26:30])
    data_start = 30 + fn_len + extra_len
    if data_start + info.compress_size > len(blob):  # unusually large extra field
        blob = reader.fetch(start, start + data_start + info.compress_size)
    data = blob[data_start : data_start + info.compress_size]

    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if len(out) != info.file_size or zlib.crc32(out) != info.CRC:
        raise ValueError(f"size/CRC mismatch for {info.filename}")
    return out
