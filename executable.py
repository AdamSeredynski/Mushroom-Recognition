from cx_Freeze import setup, Executable

setup(
    name="MushroomRecognition",
    version="0.1",
    description="",
    executables=[Executable("classify.py")],
    options={
        'build_exe': {
            'include_files': [],
            'packages': [],
            'excludes': []
        }
    }
)