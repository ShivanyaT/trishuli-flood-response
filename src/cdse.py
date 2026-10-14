"""Minimal Copernicus Data Space Ecosystem (CDSE) client: OData search + authenticated download.

Replaces `sentinelsat` from the original plan: sentinelsat targets the retired SciHub API
and does not work against CDSE.
Search needs no credentials; download needs a free CDSE account.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value"


@dataclass
class Product:
    id: str
    name: str
    start: datetime
    geom: BaseGeometry
    attrs: dict = field(default_factory=dict)
    size: int | None = None


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _parse(item: dict) -> Product:
    start = datetime.fromisoformat(item["ContentDate"]["Start"].replace("Z", "+00:00"))
    attrs = {a["Name"]: a.get("Value") for a in item.get("Attributes", [])}
    return Product(
        id=item["Id"],
        name=item["Name"].removesuffix(".SAFE"),
        start=start,
        geom=shape(item["GeoFootprint"]),
        attrs=attrs,
        size=item.get("ContentLength"),
    )


def build_filter(collection: str, aoi_wkt: str, start: datetime, end: datetime,
                 product_type: str | None = None, name_contains: str | None = None) -> str:
    parts = [f"Collection/Name eq '{collection}'"]
    if product_type:
        parts.append(
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        )
    if name_contains:
        parts.append(f"contains(Name,'{name_contains}')")
    wkt = aoi_wkt.replace("POLYGON ((", "POLYGON((").replace(", ", ",")  # CDSE docs use compact WKT
    parts.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')")
    parts.append(f"ContentDate/Start gt {_iso(start)}")
    parts.append(f"ContentDate/Start lt {_iso(end)}")
    return " and ".join(parts)


class CDSEClient:
    def __init__(self, user: str | None = None, password: str | None = None):
        self.user, self.password = user, password
        self.http = requests.Session()
        self._tok: str | None = None
        self._exp = 0.0

    # ---- search (public) -------------------------------------------------
    def search(self, collection: str, aoi_wkt: str, start: datetime, end: datetime,
               product_type: str | None = None, name_contains: str | None = None) -> list[Product]:
        params = {
            "$filter": build_filter(collection, aoi_wkt, start, end, product_type, name_contains),
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start asc",
            "$top": "200",
        }
        out: list[Product] = []
        url: str | None = CATALOGUE
        while url:
            data = self._get_json(url, params if url == CATALOGUE else None)
            out += [_parse(i) for i in data.get("value", [])]
            url = data.get("@odata.nextLink")
        log.info("%s: %d products", collection, len(out))
        return out

    def _get_json(self, url: str, params: dict | None, tries: int = 3) -> dict:
        for i in range(tries):
            try:
                r = self.http.get(url, params=params, timeout=60)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if i == tries - 1:
                    raise
                log.warning("catalogue request failed (%s); retrying", e)
                time.sleep(2 * (i + 1))
        raise RuntimeError("unreachable")

    # ---- download (needs account) ---------------------------------------
    def _token(self) -> str:
        if self._tok and time.time() < self._exp - 30:
            return self._tok
        if not (self.user and self.password):
            raise RuntimeError("CDSE_USER / CDSE_PASS not set (see .env.example)")
        r = self.http.post(TOKEN_URL, data={
            "client_id": "cdse-public", "grant_type": "password",
            "username": self.user, "password": self.password}, timeout=60)
        r.raise_for_status()
        j = r.json()
        self._tok, self._exp = j["access_token"], time.time() + int(j.get("expires_in", 600))
        return self._tok

    def download(self, p: Product, dest_dir: Path, tries: int = 5) -> Path:
        """Resumable download of a full product zip. Tokens live ~10 min, so refresh per attempt."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        final = dest_dir / f"{p.name}.zip"
        if final.exists() and (p.size is None or final.stat().st_size == p.size):
            log.info("already have %s", final.name)
            return final
        part = final.with_suffix(".zip.part")
        for attempt in range(tries):
            try:
                have = part.stat().st_size if part.exists() else 0
                headers = {"Authorization": f"Bearer {self._token()}"}
                if have:
                    headers["Range"] = f"bytes={have}-"
                url = DOWNLOAD.format(id=p.id)
                for _ in range(6):  # follow redirects manually so the auth header is kept
                    r = self.http.get(url, headers=headers, stream=True, allow_redirects=False, timeout=120)
                    if r.status_code in (301, 302, 303, 307, 308):
                        url = r.headers["Location"]
                        continue
                    break
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"
                with open(part, mode, encoding="utf-8") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                break
            except requests.RequestException as e:
                log.warning("download %s failed (%s) attempt %d/%d", p.name, e, attempt + 1, tries)
                self._tok = None
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"giving up on {p.name}")
        if p.size and part.stat().st_size != p.size:
            raise RuntimeError(f"{p.name}: size {part.stat().st_size} != expected {p.size}")
        part.rename(final)
        return final
