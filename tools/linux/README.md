# Ubuntu runtime and package

The Ubuntu version runs the same code as Windows. Only the Python runtime
(`app/python_portable_linux/`) and the launcher (`run_web_server.sh`) differ.
The runtime is not stored in git: build it on Linux, then pack it.

Target: x86_64, Ubuntu 22.04 or newer (24.04 is the reference). The wheels
need glibc >= 2.28.

## 1. Build the runtime

Run this on a Linux x86_64 machine, VM or WSL, from a checkout of the
project. It needs internet access. It does not use sudo and installs
nothing in the system.

    bash tools/linux/build_linux_runtime.sh

Work in a Linux folder (for example `~/gnss`), never in a Windows shared
folder (a VirtualBox `vboxsf` mount or `/mnt/c` in WSL): those lose the
executable bits and the symbolic links. Copy the whole project folder,
including `.git`, because `make_ubuntu_package.sh` lists the files with git.

The script:
- downloads CPython 3.11.9 from python-build-standalone (release `20240814`,
  `install_only_stripped`) and checks its SHA256;
- installs the 14 locked wheels of `requirements-linux-x86_64.txt` (hashes
  checked; the same versions as `app/python_libraries/`);
- checks the installed versions and the imports, including `app/web_server.py`.

## 2. Make the package

    bash tools/linux/make_ubuntu_package.sh [version]

This writes `dist/GNSS_Ubuntu_<version>.tar.gz` and its `.sha256` file. tar
keeps the executable bits and the symbolic links.

The package does not include `results/`, `database/`, the Windows runtime
or `run_web_server.bat`.

## 3. Use it (professors)

    tar -xzf GNSS_Ubuntu_<version>.tar.gz
    cd GNSS_Ubuntu_<version>
    ./run_web_server.sh                  # http://127.0.0.1:9000
    GNSS_WEB_PORT=9001 ./run_web_server.sh

Start the launcher from a terminal in the desktop session. It needs no sudo,
no system Python, no pip and no internet. Browse... uses zenity (or kdialog)
when installed; otherwise type the RAW path into the field.

## 4. Minimal smoke test after building

    app/python_portable_linux/bin/python3.11 --version     # Python 3.11.9
    GNSS_WEB_PORT=9123 ./run_web_server.sh &
    curl -s http://127.0.0.1:9123/ | head -c 200
    curl -s http://127.0.0.1:9123/api/paths
    curl -s -X POST -H 'Content-Type: application/json' \
         -d '{"year":2024,"doy_start":1,"doy_end":366,"th_cov":0.75}' \
         http://127.0.0.1:9123/api/ndat-modes
    ss -ltn | grep 9123                  # only 127.0.0.1:9123
    kill %1
