import ctypes, os,
d = r"C:\Users\hp\micromamba\envs\flood\Library\bin"
os.add_dll_directory(d)
for pat in ["gobject", "glib", "gio", "pango-1", "pangoft2", "harfbuzz", "fontconfig", "ffi", "intl", "pcre"]:
    print(f"{pat:11s}", [os.path.basename(p) for p in glob.glob(os.path.join(d, f"*{pat}*.dll"))])
print()
for n in ["glib-2.0-0.dll", "gobject-2.0-0.dll", "pango-1.0-0.dll", "pangoft2-1.0-0.dll"]:
    try:
        ctypes.CDLL(os.path.join(d, n)); print("OK  ", n)
    except OSError as e:
        print("FAIL", n, e)