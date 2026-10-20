"""Download CoNLL-2003 train/test files (BIO format) into ./data"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/"
os.makedirs("data", exist_ok=True)
for name in ("train.txt", "test.txt"):
    dest = os.path.join("data", name)
    print(f"Downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, dest)
print("Done.")
