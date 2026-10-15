import argparse, shutil, random, os
from pathlib import Path

def copy_pairs(imgs, labels, out_img_dir, out_lbl_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    for img in imgs:
        stem = img.stem
        lbl = img.with_suffix('.txt').parent.parent / 'labels' / img.parent.name / f'{stem}.txt'
        shutil.copy2(img, out_img_dir / img.name)
        if lbl.exists():
            shutil.copy2(lbl, out_lbl_dir / f'{stem}.txt')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images_dir', required=True, help='Root with images (all in one folder) or nested')
    p.add_argument('--labels_dir', required=True, help='Root with YOLO labels mirroring images names')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--val', type=float, default=0.2)
    p.add_argument('--test', type=float, default=0.0)
    args = p.parse_args()

    images = [p for p in Path(args.images_dir).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    # Move labels into a parallel structure in memory
    # For simplicity, we assume label filenames match image stems.
    random.shuffle(images)
    n = len(images)
    n_test = int(n * args.test)
    n_val = int(n * args.val)
    test_imgs = images[:n_test]
    val_imgs = images[n_test:n_test+n_val]
    train_imgs = images[n_test+n_val:]

    out = Path(args.out_dir)
    for split, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        out_img = out / 'images' / split
        out_lbl = out / 'labels' / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            stem = img.stem
            lbl = Path(args.labels_dir) / f'{stem}.txt'
            shutil.copy2(img, out_img / img.name)
            if lbl.exists():
                shutil.copy2(lbl, out_lbl / f'{stem}.txt')

    print(f'[✓] Split complete -> {out}')

if __name__ == '__main__':
    main()
