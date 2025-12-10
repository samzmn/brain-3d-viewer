@echo off
setlocal

REM 1. Activate venv (adjust path if different)
call .venv\Scripts\activate

REM 2. Clean previous builds
rmdir /s /q build
rmdir /s /q dist
del /q BrainViewer.spec

REM 3. Run PyInstaller (spec if exists, otherwise generate)
if exist BrainViewer.spec (
    python -m PyInstaller BrainViewer.spec
) else (
    pyinstaller --name BrainViewer --windowed --onedir --paths src src\main.py
)

REM 4. Package source zip
powershell -Command "Compress-Archive -Path src/*, README.md, licenses/* -DestinationPath source_code.zip -Force"

REM 5. Build Inno Setup installer
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" setup.iss

echo Build complete.
pause
