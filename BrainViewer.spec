# BrainViewer.spec
# PyInstaller spec for BrainViewer
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

project_root = Path.cwd()
src_dir = project_root / "src"

# collect any dynamic imports in your code (if needed)
# hiddenimports = collect_submodules('numpy', 'scipy', 'nibabel', 'PyQt5', 'matplotlib', 'scikit-image')  # example; remove if not needed
hiddenimports = []
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('nibabel')
hiddenimports += collect_submodules('PyQt5')
hiddenimports += collect_submodules('matplotlib')

# Data files to include: (source, target_relative_dir)
datas = [
    (str(project_root / "licenses"), "licenses"),            # include license folder
    # (str(src_dir / "resources"), "resources"),              # your app resources (icons, ui files)
    # add other files: e.g. (r"src\some_data.nii", ".")
]

# -------------------------------------------------------------------
# Include NumPy and SciPy binary libraries (.dll)
# -------------------------------------------------------------------
import glob

binaries = []

# numpy/.libs
numpy_lib_path = Path(sys.exec_prefix) / "Lib" / "site-packages" / "numpy" / ".libs"
if numpy_lib_path.exists():
    for f in glob.glob(str(numpy_lib_path / "*")):
        binaries.append((f, "."))

# scipy/.libs (optional, only if you use scipy.ndimage)
scipy_lib_path = Path(sys.exec_prefix) / "Lib" / "site-packages" / "scipy" / ".libs"
if scipy_lib_path.exists():
    for f in glob.glob(str(scipy_lib_path / "*")):
        binaries.append((f, "."))

a = Analysis(
    [str(src_dir / "main.py")],
    pathex=[str(src_dir), str(project_root)],   # ensure src on path
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name="BrainViewer",
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False  # windowed application
          )

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="BrainViewer"
)
