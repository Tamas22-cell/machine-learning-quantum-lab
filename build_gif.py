import os
import zipfile
import shutil
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    print("HIBA: Pillow nincs telepítve. Következő lépésben telepítjük.")
    raise

# ===== BEÁLLÍTÁS =====
ZIP_NAME = "ml-qml-8kep.zip"
OUT_NAME = "ml-qml-linkedin.gif"

# LinkedIn-barát 4:5 arány
OUT_W = 800
OUT_H = 1000

# animáció
FPS = 10
HOLD_FRAMES = 10
FADE_FRAMES = 4
MAX_ZOOM = 1.08  # finom zoom
BG_COLOR = (8, 17, 33)  # sötét háttér


def find_zip():
    here = Path.cwd()
    zip_path = here / ZIP_NAME
    if zip_path.exists():
        return zip_path

    # fallback: első zip a mappában
    zips = list(here.glob("*.zip"))
    if zips:
        return zips[0]

    raise FileNotFoundError(
        f"Nem találom a ZIP-et ebben a mappában: {here}\n"
        f"Tedd ide a fájlt ezzel a névvel: {ZIP_NAME}"
    )


def extract_images(zip_path: Path):
    extract_dir = Path.cwd() / "_gif_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = []

    for p in extract_dir.rglob("*"):
        if p.suffix.lower() in image_exts:
            images.append(p)

    images = sorted(images)
    if not images:
        raise RuntimeError("Nem találtam képeket a ZIP-ben.")

    return images, extract_dir


def fit_canvas(img: Image.Image, w: int, h: int):
    bg = Image.new("RGB", (w, h), BG_COLOR)
    fitted = ImageOps.contain(img, (w - 30, h - 30), Image.Resampling.LANCZOS)
    x = (w - fitted.width) // 2
    y = (h - fitted.height) // 2
    bg.paste(fitted, (x, y))
    return bg


def ken_burns_frames(base_img: Image.Image, count: int):
    frames = []

    for i in range(count):
        t = i / max(1, count - 1)

        scale = 1.0 + (MAX_ZOOM - 1.0) * t
        nw = int(base_img.width * scale)
        nh = int(base_img.height * scale)

        zoomed = base_img.resize((nw, nh), Image.Resampling.LANCZOS)

        # finom drift
        drift_x = int((nw - base_img.width) * 0.15 * t)
        drift_y = int((nh - base_img.height) * 0.20 * (1 - t))

        left = max(0, min(nw - base_img.width, (nw - base_img.width) // 2 + drift_x))
        top = max(0, min(nh - base_img.height, (nh - base_img.height) // 2 + drift_y))

        crop = zoomed.crop((left, top, left + base_img.width, top + base_img.height))
        frames.append(crop)

    return frames


def build_gif(images):
    prepared = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        prepared.append(fit_canvas(img, OUT_W, OUT_H))

    frames = []

    for idx, img in enumerate(prepared):
        motion = ken_burns_frames(img, HOLD_FRAMES)
        frames.extend(motion)

        if idx < len(prepared) - 1:
            a = motion[-1]
            b = prepared[idx + 1]

            for j in range(1, FADE_FRAMES + 1):
                alpha = j / (FADE_FRAMES + 1)
                blend = Image.blend(a, b, alpha)
                frames.append(blend)

    # loop lezárás: utolsó -> első finom átmenet
    last_img = prepared[-1]
    first_img = prepared[0]
    for j in range(1, FADE_FRAMES + 1):
        alpha = j / (FADE_FRAMES + 1)
        blend = Image.blend(last_img, first_img, alpha)
        frames.append(blend)

    out_path = Path.cwd() / OUT_NAME
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )

    return out_path


def main():
    print("GIF készítés indul...")
    zip_path = find_zip()
    print(f"ZIP: {zip_path}")

    images, extract_dir = extract_images(zip_path)
    print(f"Képek száma: {len(images)}")

    out_path = build_gif(images)
    print(f"KÉSZ: {out_path}")

    # takarítás
    shutil.rmtree(extract_dir, ignore_errors=True)


if __name__ == "__main__":
    main()