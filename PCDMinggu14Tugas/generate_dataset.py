"""
Generate synthetic dataset: 6 classes (cat, dog, bird, car, truck, motorcycle)
Each class: 150 images (32x32 RGB)
"""
import numpy as np
from PIL import Image, ImageDraw
import os, random

random.seed(42)
np.random.seed(42)

def draw_cat(size=32):
    img = Image.new('RGB', (size, size), color=(200, 180, 160))
    d = ImageDraw.Draw(img)
    s = size
    # body
    d.ellipse([s*0.2, s*0.35, s*0.8, s*0.85], fill=(150,120,90))
    # head
    d.ellipse([s*0.25, s*0.1, s*0.75, s*0.5], fill=(160,130,100))
    # ears
    d.polygon([(s*0.25,s*0.25),(s*0.15,s*0.05),(s*0.38,s*0.18)], fill=(140,110,80))
    d.polygon([(s*0.75,s*0.25),(s*0.85,s*0.05),(s*0.62,s*0.18)], fill=(140,110,80))
    # eyes
    d.ellipse([s*0.33,s*0.22,s*0.43,s*0.32], fill=(50,80,50))
    d.ellipse([s*0.57,s*0.22,s*0.67,s*0.32], fill=(50,80,50))
    # tail
    d.arc([s*0.6,s*0.5,s*0.95,s*0.9], 0, 270, fill=(150,120,90), width=3)
    return img

def draw_dog(size=32):
    img = Image.new('RGB', (size, size), color=(220, 210, 190))
    d = ImageDraw.Draw(img)
    s = size
    d.ellipse([s*0.15, s*0.35, s*0.85, s*0.85], fill=(180,140,100))
    d.ellipse([s*0.2, s*0.08, s*0.8, s*0.48], fill=(190,150,110))
    # floppy ears
    d.ellipse([s*0.05,s*0.15,s*0.3,s*0.55], fill=(160,120,80))
    d.ellipse([s*0.7,s*0.15,s*0.95,s*0.55], fill=(160,120,80))
    d.ellipse([s*0.32,s*0.2,s*0.44,s*0.32], fill=(40,30,20))
    d.ellipse([s*0.56,s*0.2,s*0.68,s*0.32], fill=(40,30,20))
    d.arc([s*0.35,s*0.32,s*0.65,s*0.45], 0, 180, fill=(100,60,40), width=2)
    return img

def draw_bird(size=32):
    img = Image.new('RGB', (size, size), color=(180, 210, 230))
    d = ImageDraw.Draw(img)
    s = size
    d.ellipse([s*0.2, s*0.35, s*0.75, s*0.75], fill=(70,130,180))
    d.ellipse([s*0.35, s*0.1, s*0.7, s*0.42], fill=(70,130,180))
    # wing
    d.ellipse([s*0.05, s*0.35, s*0.45, s*0.65], fill=(50,100,160))
    # beak
    d.polygon([(s*0.7,s*0.22),(s*0.9,s*0.27),(s*0.7,s*0.32)], fill=(255,200,0))
    d.ellipse([s*0.55,s*0.15,s*0.65,s*0.25], fill=(20,20,20))
    # legs
    d.line([(s*0.4,s*0.75),(s*0.35,s*0.92)], fill=(255,200,0), width=2)
    d.line([(s*0.55,s*0.75),(s*0.6,s*0.92)], fill=(255,200,0), width=2)
    return img

def draw_car(size=32):
    img = Image.new('RGB', (size, size), color=(200, 200, 220))
    d = ImageDraw.Draw(img)
    s = size
    d.rectangle([s*0.05,s*0.5,s*0.95,s*0.82], fill=(60,100,180))
    d.rectangle([s*0.2,s*0.3,s*0.8,s*0.55], fill=(80,120,200))
    # windows
    d.rectangle([s*0.23,s*0.33,s*0.47,s*0.5], fill=(180,220,255))
    d.rectangle([s*0.53,s*0.33,s*0.77,s*0.5], fill=(180,220,255))
    # wheels
    d.ellipse([s*0.08,s*0.68,s*0.32,s*0.92], fill=(30,30,30))
    d.ellipse([s*0.68,s*0.68,s*0.92,s*0.92], fill=(30,30,30))
    d.ellipse([s*0.13,s*0.73,s*0.27,s*0.87], fill=(80,80,80))
    d.ellipse([s*0.73,s*0.73,s*0.87,s*0.87], fill=(80,80,80))
    return img

def draw_truck(size=32):
    img = Image.new('RGB', (size, size), color=(210, 200, 180))
    d = ImageDraw.Draw(img)
    s = size
    # cargo
    d.rectangle([s*0.3,s*0.25,s*0.95,s*0.78], fill=(180,100,60))
    # cab
    d.rectangle([s*0.05,s*0.38,s*0.35,s*0.78], fill=(200,120,80))
    d.rectangle([s*0.08,s*0.4,s*0.3,s*0.55], fill=(180,220,255))
    # wheels
    d.ellipse([s*0.05,s*0.65,s*0.25,s*0.88], fill=(30,30,30))
    d.ellipse([s*0.55,s*0.65,s*0.75,s*0.88], fill=(30,30,30))
    d.ellipse([s*0.72,s*0.65,s*0.92,s*0.88], fill=(30,30,30))
    return img

def draw_motorcycle(size=32):
    img = Image.new('RGB', (size, size), color=(210, 210, 200))
    d = ImageDraw.Draw(img)
    s = size
    # wheels
    d.ellipse([s*0.02,s*0.5,s*0.38,s*0.92], outline=(30,30,30), width=3)
    d.ellipse([s*0.62,s*0.5,s*0.98,s*0.92], outline=(30,30,30), width=3)
    d.ellipse([s*0.08,s*0.56,s*0.32,s*0.86], fill=(80,80,80))
    d.ellipse([s*0.68,s*0.56,s*0.92,s*0.86], fill=(80,80,80))
    # body / frame
    d.line([(s*0.2,s*0.7),(s*0.5,s*0.4),(s*0.8,s*0.7)], fill=(150,30,30), width=3)
    d.line([(s*0.5,s*0.4),(s*0.5,s*0.2),(s*0.7,s*0.2)], fill=(100,100,100), width=2)
    # seat
    d.rectangle([s*0.35,s*0.3,s*0.65,s*0.42], fill=(40,40,40))
    return img

CLASSES = {
    'animals/cat': draw_cat,
    'animals/dog': draw_dog,
    'animals/bird': draw_bird,
    'vehicles/car': draw_car,
    'vehicles/truck': draw_truck,
    'vehicles/motorcycle': draw_motorcycle,
}

base = '/home/claude/cnn_project/dataset'
N = 150
for cls, fn in CLASSES.items():
    path = f'{base}/{cls}'
    os.makedirs(path, exist_ok=True)
    for i in range(N):
        img = fn(32)
        # Add noise for variation
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15 + i % 20, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        # Random brightness
        factor = 0.7 + random.random() * 0.6
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(f'{path}/img_{i:03d}.png')

total = sum(len(os.listdir(f'{base}/{c}')) for c in CLASSES)
print(f"Dataset generated: {total} images across {len(CLASSES)} classes")
for cls in CLASSES:
    print(f"  {cls}: {len(os.listdir(f'{base}/{cls}'))} images")
