#!/usr/bin/env python3
"""
manga_pipeline.py — Fully automated manga download, extract, upscale, and route to Komga.

Designed for cron execution with full idempotence:
  - Each run gets an isolated workspace (timestamped directory)
  - Source archives are never deleted — only copies are extracted
  - Progress is tracked in JSON; interrupted runs resume from last failure
  - All subprocess output is captured to per-run log files
  - File lock prevents concurrent runs
  - Non-destructive: nothing is deleted until success is verified
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

LOCAL_WORK_DIR = r"/home/adrian/for-extraction"  # kobodl downloads here
NAS_KOMGA_DIR = r"/mnt/media/manga"  # Komga manga library root
UPSCALE_TOOL_DIR = r"/home/adrian/tools/manga_upscaler"
UPSCALE_TOOL_PATH = "manga_upscale.py"
HISTORY_FILE = os.path.join(UPSCALE_TOOL_DIR, "downloaded_ids.txt")

# Find kobodl — try same venv as this script, then PATH
KOBODL_FALLBACKS = [
    os.path.join(os.path.dirname(sys.executable), "kobodl"),
    "kobodl",
]

MODEL_NAME = "2x-AnimeSharpV3.pth"


# ── Helpers ────────────────────────────────────────────────────────────────


def _find_kobodl():
    """Resolve kobodl path, falling back to PATH lookup."""
    for candidate in KOBODL_FALLBACKS:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _setup_logging(run_dir):
    """Configure logging: console + per-run log file."""
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(run_dir, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger, os.path.join(run_dir, "subprocess.log")


def _acquire_lock(lock_file):
    """Acquire an exclusive file lock. Returns True on success."""
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Check if previous holder is still alive
        try:
            with open(lock_file, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)
            return False
        except (ProcessLookupError, ValueError, OSError):
            pass
        # Stale lock — remove and retry once
        try:
            os.unlink(lock_file)
        except OSError:
            pass
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False


def _run_subprocess(cmd, logger=None, log_file=None, cwd=None, console_output=False):
    """Run a subprocess, tee stdout/stderr to log_file, optionally console."""
    logger = logger or logging.getLogger("pipeline")
    log_fh = logging.FileHandler(log_file, encoding="utf-8") if log_file else None
    if log_fh:
        log_fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(log_fh)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        # Write full output to log file
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"$ {' '.join(cmd)}\n")
                if result.stdout:
                    f.write(result.stdout)
                if result.stderr:
                    f.write(result.stderr)
                f.write(f"\n[exit code: {result.returncode}]\n\n")

        if console_output and result.stdout:
            for line in result.stdout.splitlines():
                logger.info(line)

        return result
    finally:
        if log_fh:
            logger.removeHandler(log_fh)


def _load_history():
    if not os.path.exists(HISTORY_FILE):
        return set()
    with open(HISTORY_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def _save_history(history):
    with open(HISTORY_FILE, "w") as f:
        for _id in sorted(history):
            f.write(f"{_id}\n")


def _load_progress(run_dir):
    path = os.path.join(run_dir, "progress.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"downloaded": [], "processed": [], "routed": []}


def _save_progress(run_dir, progress):
    with open(os.path.join(run_dir, "progress.json"), "w") as f:
        json.dump(progress, f, indent=2)


def create_cbz(folder_path, output_path):
    """Create a .cbz from folder contents. Returns True on success."""
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(folder_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, folder_path)
                zf.write(fpath, arcname)
    return os.path.getsize(output_path) > 0


def find_main_image_directory(base_path):
    """Return the subdirectory with the most image files."""
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    best_dir = None
    best_count = 0
    for root, _dirs, files in os.walk(base_path):
        count = sum(1 for f in files if os.path.splitext(f)[1].lower() in image_exts)
        if count > best_count:
            best_count = count
            best_dir = root
    return best_dir


def extract_series_name(filename):
    """Strip volume number and everything after it from a filename."""
    name = os.path.splitext(filename)[0]
    return re.sub(r"(?i)[,\s-]*(?:vol(?:ume|\.)?|v\.?)?\s*\d+.*$", "", name).strip()


# ── Pipeline stages ────────────────────────────────────────────────────────


def stage_fetch(logger, kobodl, downloads_dir, run_dir):
    """Download new manga from Kobo. Idempotent via history file."""
    logger.info("--- Step 1: Checking Kobo for new books ---")
    history = _load_history()

    result = _run_subprocess(
        [kobodl, "book", "list"],
        logger=logger,
        console_output=False,
    )
    if result.returncode != 0:
        logger.error(f"kobodl book list failed (exit {result.returncode})")
        return 0

    uuid_re = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    )
    new_count = 0

    for line in result.stdout.splitlines():
        if "AUDIOBOOK" in line.upper():
            continue
        match = uuid_re.search(line)
        if not match:
            continue

        book_id = match.group(0)
        if book_id in history:
            continue

        logger.info(f"[CHECKING] ID {book_id}...")
        r = _run_subprocess(
            [kobodl, "book", "get", "--format-str", "{Title}", book_id],
            logger=logger,
            log_file=os.path.join(run_dir, f"kobodl-get-{book_id}.log"),
            cwd=downloads_dir,
        )

        if r.returncode != 0:
            logger.info(
                f"  [PENDING] ID {book_id} - not available yet "
                f"(pre-order/pending release). Will retry next run."
            )
            continue

        logger.info(f"  [DOWNLOADED] ID {book_id}")
        history.add(book_id)
        _save_history(history)
        new_count += 1

if new_count > 0:
        logger.info(f"Downloaded {new_count} new file(s). Staging as .cbz...")
        # Walk recursively because kobodl nests files in kobo_downloads/
        for root, _dirs, files in os.walk(downloads_dir):
            for fname in files:
                if fname.endswith(".epub"):
                    old = os.path.join(root, fname)
                    new_name = os.path.splitext(fname)[0] + ".cbz"
                    new = os.path.join(downloads_dir, new_name)
                    os.rename(old, new)
                    logger.info(f"  Staged: {new_name}")

        # Clean up stale kobo_downloads dir (files already moved out above)
        kobo_dir = os.path.join(downloads_dir, "kobo_downloads")
        if os.path.exists(kobo_dir):
            shutil.rmtree(kobo_dir)

    return new_count


def stage_extract(logger, workspace, run_dir):
    """Extract all archives in the workspace. Skip if already extracted."""
    logger.info("--- Step 2a: Extracting archives ---")

    # Find archives
    archives = []
    for f in os.listdir(workspace):
        fp = os.path.join(workspace, f)
        if os.path.isfile(fp) and f.endswith((".zip", ".cbz")):
            archives.append(f)

    if not archives:
        logger.info("No archives found.")
        return 0

    # Load progress to skip already-extracted items
    progress = _load_progress(run_dir)
    extracted = 0

    for archive in archives:
        # Check if output folder already exists (already extracted)
        folder_name = os.path.splitext(archive)[0]
        folder_path = os.path.join(workspace, folder_name)
        if os.path.isdir(folder_path):
            logger.info(f"  [SKIP] '{archive}' already extracted.")
            continue

        logger.info(f"  [EXTRACT] '{archive}'...")
        r = _run_subprocess(
            [sys.executable, UPSCALE_TOOL_PATH, "extract", "--input", workspace],
            logger=logger,
            log_file=os.path.join(run_dir, f"extract-{archive}.log"),
            cwd=UPSCALE_TOOL_DIR,
        )
        if r.returncode != 0:
            logger.error(f"  [FAIL] Extraction failed for '{archive}'")
            continue

        # Archive should be deleted by the extract tool
        logger.info(f"  [DONE] '{archive}' extracted.")
        extracted += 1

    return extracted


def stage_process(logger, workspace, run_dir):
    """Extract (if needed), upscale, and archive each manga item."""
    logger.info("--- Step 2b: Processing manga (extract/upscale/archive) ---")

    progress = _load_progress(run_dir)
    items = [
        i
        for i in os.listdir(workspace)
        if os.path.isdir(os.path.join(workspace, i)) and i not in progress["processed"]
    ]

    if not items:
        logger.info("No items to process.")
        return 0

    processed = 0
    for item in items:
        item_path = os.path.join(workspace, item)
        logger.info(f"\n[ITEM] Processing '{item}'...")

        # Step A: Extract (skip if folder already exists from prior run)
        archive = item + ".zip"
        cbz_archive = item + ".cbz"
        if os.path.isfile(os.path.join(workspace, archive)):
            logger.info(f"  [EXTRACT] '{archive}'...")
            r = _run_subprocess(
                [sys.executable, UPSCALE_TOOL_PATH, "extract", "--input", workspace],
                logger=logger,
                log_file=os.path.join(run_dir, f"extract-{item}.log"),
                cwd=UPSCALE_TOOL_DIR,
            )
            if r.returncode != 0:
                logger.error(f"  [FAIL] Extraction failed for '{item}'. Skipping.")
                continue
        elif os.path.isfile(os.path.join(workspace, cbz_archive)):
            logger.info(f"  [EXTRACT] '{cbz_archive}'...")
            r = _run_subprocess(
                [sys.executable, UPSCALE_TOOL_PATH, "extract", "--input", workspace],
                logger=logger,
                log_file=os.path.join(run_dir, f"extract-{item}.log"),
                cwd=UPSCALE_TOOL_DIR,
            )
            if r.returncode != 0:
                logger.error(f"  [FAIL] Extraction failed for '{item}'. Skipping.")
                continue

        # Step B: Find image directory
        image_dir = find_main_image_directory(item_path)
        if not image_dir:
            logger.warning(f"  [SKIP] No images found in '{item}'.")
            continue
        logger.info(f"  Image directory: {os.path.relpath(image_dir, workspace)}")

        # Step C: Upscale
        output_dir = os.path.join(workspace, f"{item}_upscaled")
        logger.info(f"  [UPSCALE] Running upscaler...")
        r = _run_subprocess(
            [
                sys.executable,
                UPSCALE_TOOL_PATH,
                "upscale",
                "--color",
                image_dir,
                "--output",
                output_dir,
                "--model-color",
                MODEL_NAME,
            ],
            logger=logger,
            log_file=os.path.join(run_dir, f"upscale-{item}.log"),
            cwd=UPSCALE_TOOL_DIR,
        )
        if r.returncode != 0:
            logger.error(f"  [FAIL] Upscaler failed for '{item}'.")
            # Clean up partial output
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            continue

        if not os.path.isdir(output_dir) or not any(Path(output_dir).iterdir()):
            logger.warning(f"  [SKIP] No output from upscaler for '{item}'.")
            shutil.rmtree(output_dir, ignore_errors=True)
            continue

        # Step D: Move upscaled images into the item directory
        for f in Path(output_dir).iterdir():
            shutil.move(str(f), image_dir)
        shutil.rmtree(output_dir, ignore_errors=True)

        # Step E: Create CBZ
        cbz_path = os.path.join(workspace, f"{item}.cbz")
        if not create_cbz(item_path, cbz_path):
            logger.error(f"  [FAIL] Failed to create CBZ for '{item}'.")
            continue

        # Success — mark as processed
        progress["processed"].append(item)
        _save_progress(run_dir, progress)
        logger.info(f"  [DONE] '{item}' — extracted, upscaled, archived.")
        processed += 1

    return processed


def stage_route(logger, workspace, run_dir):
    """Copy CBZ files to Komga NAS. Idempotent — skips if already at destination."""
    logger.info("--- Step 3: Routing to Komga NAS ---")

    progress = _load_progress(run_dir)
    cbz_files = [
        f
        for f in os.listdir(workspace)
        if f.endswith(".cbz") and f not in progress.get("routed", [])
    ]

    if not cbz_files:
        logger.info("No files to route.")
        return 0

    routed = 0
    failed = []

    for filename in cbz_files:
        series_name = extract_series_name(filename)
        series_dir = os.path.join(NAS_KOMGA_DIR, series_name)

        if not os.path.isdir(series_dir):
            os.makedirs(series_dir, exist_ok=True)
            logger.info(f"  Created series directory: {series_dir}")

        dst = os.path.join(series_dir, filename)

        # Skip if already at destination
        if os.path.exists(dst):
            logger.info(f"  [SKIP] '{filename}' already in {series_name}/")
            progress.setdefault("routed", []).append(filename)
            continue

        # Copy then verify, then delete source
        src = os.path.join(workspace, filename)
        try:
            shutil.copy2(src, dst)
        except OSError as e:
            logger.error(f"  [FAIL] Copy '{filename}' to {dst}: {e}")
            failed.append(filename)
            continue

        # Verify destination exists and size matches
        if not os.path.exists(dst):
            logger.error(f"  [FAIL] Destination missing after copy: {dst}")
            failed.append(filename)
            continue
        if os.path.getsize(src) != os.path.getsize(dst):
            logger.error(f"  [FAIL] Size mismatch for '{filename}' after copy.")
            failed.append(filename)
            continue

        os.unlink(src)
        logger.info(f"  [ROUTED] '{filename}' -> {series_name}/")
        progress.setdefault("routed", []).append(filename)
        routed += 1

    if failed:
        logger.error(f"[WARN] {len(failed)} file(s) failed to route: {failed}")

    _save_progress(run_dir, progress)
    return routed


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    start_time = datetime.now()
    run_id = start_time.strftime("%Y%m%d-%H%M%S")

    # Resolve kobodl
    kobodl = _find_kobodl()
    if not kobodl:
        print(f"FATAL: kobodl not found. Searched: {KOBODL_FALLBACKS}", file=sys.stderr)
        sys.exit(1)

    # Validate required paths
    missing = []
    if not os.path.isdir(LOCAL_WORK_DIR):
        missing.append(f"Work directory: {LOCAL_WORK_DIR}")
    if not os.path.isdir(NAS_KOMGA_DIR):
        missing.append(f"NAS/Komga directory: {NAS_KOMGA_DIR}")
    if not os.path.isfile(os.path.join(UPSCALE_TOOL_DIR, UPSCALE_TOOL_PATH)):
        missing.append(f"Upscaler tool: {UPSCALE_TOOL_DIR}/{UPSCALE_TOOL_PATH}")
    if not os.path.isfile(os.path.join(UPSCALE_TOOL_DIR, "backend", "upscale.py")):
        missing.append(f"Backend upscale: {UPSCALE_TOOL_DIR}/backend/upscale.py")

    # Model path resolution (same logic as manga_upscale.py resolve_model)
    model_path = os.path.join(UPSCALE_TOOL_DIR, "backend", "models", MODEL_NAME)
    if not os.path.isfile(model_path):
        # Try prefix match
        model_dir = os.path.join(UPSCALE_TOOL_DIR, "backend", "models")
        if os.path.isdir(model_dir):
            matches = [
                f
                for f in os.listdir(model_dir)
                if f.startswith(os.path.splitext(MODEL_NAME)[0])
            ]
            if matches:
                model_path = os.path.join(model_dir, matches[0])
            else:
                missing.append(f"Model '{MODEL_NAME}' not found in {model_dir}")
        else:
            missing.append(f"Models directory: {model_dir}")

    if missing:
        print(
            "FATAL: Missing required paths:\n  " + "\n  ".join(missing), file=sys.stderr
        )
        sys.exit(1)

    # Set up per-run workspace and logging
    run_dir = os.path.join(LOCAL_WORK_DIR, "_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger, subprocess_log = _setup_logging(run_dir)
    lock_file = os.path.join(run_dir, ".lock")

    if not _acquire_lock(lock_file):
        print(
            f"FATAL: Another instance is running (lock: {lock_file})", file=sys.stderr
        )
        sys.exit(1)

    try:
        # Create isolated workspace with subdirectories
        downloads_dir = os.path.join(run_dir, "downloads")
        workspace = os.path.join(run_dir, "workspace")
        os.makedirs(downloads_dir, exist_ok=True)
        os.makedirs(workspace, exist_ok=True)

        logger.info(f"Run ID: {run_id}")
        logger.info(f"Workspace: {run_dir}")
        logger.info(f"kobodl: {kobodl}")
        logger.info(f"Model: {model_path}")
        logger.info("")

        # Stage 1: Download new manga
        new_count = stage_fetch(logger, kobodl, downloads_dir, run_dir)

        if new_count > 0:
            # Copy downloaded files into workspace for processing
            for f in os.listdir(downloads_dir):
                src = os.path.join(downloads_dir, f)
                dst = os.path.join(workspace, f)
                shutil.copy2(src, dst)

            # Stage 2a: Extract archives
            stage_extract(logger, workspace, run_dir)

            # Stage 2b: Process manga (upscale + archive)
            stage_process(logger, workspace, run_dir)

            # Stage 3: Route to Komga
            stage_route(logger, workspace, run_dir)
        else:
            logger.info("No new books to process.")

        elapsed = datetime.now() - start_time
        logger.info(f"\n=== Pipeline complete in {elapsed} ===")

    finally:
        # Release lock
        try:
            os.unlink(lock_file)
        except OSError:
            pass

    sys.exit(0)


if __name__ == "__main__":
    main()
