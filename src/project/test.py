from pathlib import Path

# Målet for symlinket (behøver IKKE at eksistere)
c = Path("hello/med/dig")

# Den mappe, som skal indeholde symlinket
Path("hej").mkdir(exist_ok=True)   # <-- DETTE manglede før ✅

# Stien til symlinket
p = Path("hej/far")

# Opret symlink: hej/far -> hello/med/dig
p.symlink_to(c, target_is_directory=True)
