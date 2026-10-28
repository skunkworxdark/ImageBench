# ImageBench 🗜️

A hardware-native, real-time image compression benchmarking dashboard and sandbox built with [Streamlit](https://streamlit.io/) and [Pillow](https://python-pillow.org/).

ImageBench evaluates image encoders and decoders on your actual machine using real runtime operations. It completely avoids synthetic mathematical approximations, simulated size heuristics, or timing placeholders.

---

## Key Features

- **Strict Hardware-Native Metrics:**
  - **File Size (MiB):** Measured with `os.path.getsize()` from real files written to disk.
  - **Save / Encode Latency (ms):** Measured by wrapping the actual encoder call and disk write tightly within `time.perf_counter()`.
  - **Load / Decode Latency (ms):** Measured by reading the file and forcing a full raster unpack into RAM (`im.load()`).
  - **Untimed Source Normalization:** Ingested images are normalized into a canonical uncompressed 8-bit RGB/RGBA raster once, ensuring every codec processes the exact same raw pixel payload.
  - **Library Warm-up:** An untimed warm-up pass primes codec DLLs and lookup tables so initial library overhead is never billed to user measurements.

- **Supported Codecs & Engines:**
  - **PNG [PIL NATIVE]:** Lossless compression with configurable `compress_level` (0–9).
  - **WebP Lossless [PIL NATIVE]:** True lossless mode with configurable `method` (0–6).
  - **WebP Lossy [PIL NATIVE]:** Configurable quality (50–100) and compression `method` (0–6).
  - **JPEG XL Lossless [PIL NATIVE]:** Powered by `pillow-jxl-plugin` with configurable `effort` (1–9) and `decoding_speed` (0–4).
  - **JPEG XL Lossy [PIL NATIVE]:** Powered by `pillow-jxl-plugin` with configurable `quality` (50–100), `effort` (1–9), and `decoding_speed` (0–4).

- **Interactive Sandbox & Analytics:**
  - **Live Dynamic Weight Slider:** Adjust the Storage Size vs. Save Latency trade-off (0% to 100%) in real time. The leaderboard re-sorts and updates composite scores instantly without re-encoding any image.
  - **Configurable Codec Matrix:** Enable or disable entire codec families or individual tiers from the sidebar. Disabled profiles are cleanly stripped from the UI while keeping their cached measurements.
  - **Lossless Architecture Leaderboard:** Normalizes disparate units (milliseconds vs. megabytes) onto a 0–100 scale anchored against the worst-performing baseline to produce an objective composite score.
  - **Lossy Archive Transformation Index:** Side-by-side comparison of lossy contenders (e.g. WebP Lossy vs. JPEG XL Lossy) detailing space reclaimed vs. original files, bits per pixel, and encoding/decoding speed trade-offs.
  - **Aggregated Metrics Ledger:** Running Average, Minimum, and Maximum for file sizes, save latency, and load latency across all active images.
  - **Performance Spike Log:** Automatic alerting when an image's save latency or file size exceeds 200% of the active tier average.
  - **Selective Dataset Toggles:** Check or uncheck individual files to modify aggregate calculations dynamically.
  - **CSV Export:** Download full per-file and per-profile benchmark data with one click.
  - **Workspace Reset:** Flush all temporary files, cache, and session state.

---

## Quick Start

### Windows

Run the automated scripts:

```cmd
setup.bat
run.bat
```

* `setup.bat` creates a `.venv` (using Python 3.13, 3.12, or 3.11) and installs all dependencies.
* `run.bat` boots the Streamlit dashboard in your default browser.

### macOS & Linux

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

### Manual Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

---

## Dependencies

- Python 3.11+
- `streamlit`
- `pillow`
- `numpy`
- `pillow-jxl-plugin`

---

## License

This project is licensed under the [MIT License](LICENSE).
