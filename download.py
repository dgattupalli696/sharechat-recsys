import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
import urllib.request


ROOT = Path(__file__).resolve().parent
API = "https://www.kaggle.com/api/v1/competitions"


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def main():
    token = (Path.home() / ".kaggle/access_token").read_text().strip()
    opener = urllib.request.build_opener(SafeRedirect())

    def request(url):
        return opener.open(
            urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
            timeout=120,
        )

    with request(f"{API}/data/list/sharechat") as response:
        manifest = json.load(response)
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        name = item["name"]
        if Path(name).name != name:
            raise ValueError(f"Unexpected filename: {name!r}")
        destination = data / name
        expected = item["totalBytes"]
        if destination.exists() and destination.stat().st_size == expected:
            print(f"Already downloaded: {name}", flush=True)
            continue
        partial = destination.with_suffix(destination.suffix + ".partial")
        print(f"Downloading {name}: {expected / 1e9:.2f} GB", flush=True)
        with request(f"{API}/data/download/sharechat/{name}") as response:
            signed_url = response.url
        if urlparse(signed_url).hostname == "www.kaggle.com":
            raise RuntimeError("Expected a signed storage download URL")

        def fetch_range(bounds):
            start, end = bounds
            req = urllib.request.Request(
                signed_url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError("Server did not honor the range request")
                if response.headers.get("Content-Range") != (
                    f"bytes {start}-{end}/{expected}"
                ):
                    raise RuntimeError("Unexpected response byte range")
                content = response.read()
            if len(content) != end - start + 1:
                raise RuntimeError("Incomplete download range")
            return content

        total = partial.stat().st_size if partial.exists() else 0
        if total > expected:
            raise ValueError(f"Oversized partial download: {partial}")
        chunk_size = 32 * 1024 * 1024
        with partial.open("ab") as output, ThreadPoolExecutor(max_workers=4) as pool:
            last_report = total
            while total < expected:
                ranges = [
                    (start, min(start + chunk_size, expected) - 1)
                    for start in range(
                        total, min(total + 4 * chunk_size, expected), chunk_size
                    )
                ]
                for content in pool.map(fetch_range, ranges):
                    output.write(content)
                    total += len(content)
                output.flush()
                if total - last_report >= 512 * 1024 * 1024 or total == expected:
                    print(f"  {name}: {total / expected:.0%}", flush=True)
                    last_report = total
        if total != expected:
            raise ValueError(f"{name}: expected {expected} bytes, got {total}")
        os.replace(partial, destination)
        print(f"Downloaded {name}", flush=True)


if __name__ == "__main__":
    main()
