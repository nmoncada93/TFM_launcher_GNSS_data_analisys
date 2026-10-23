#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Everything the web server does that depends on the operating system:
choosing a folder in a native dialog and opening a folder in the file
manager. Nothing scientific lives here, and the Python runtime is chosen by
the launchers, not here.

    choose_directory(title, initial_dir=None) -> absolute path, or None if cancelled
    open_folder(path)                         -> None
    open_browser(url)                         -> True if a browser was asked to open it

choose_directory and open_folder raise one of three errors, which
web_server.py turns into HTTP codes (open_browser never raises):

    ActionUnavailableError  nothing usable on this computer       -> 501
    ActionFailedError       a dialog/file manager started and failed -> 500
    ActionTimeoutError      no answer within DIALOG_TIMEOUT_S      -> 504

Windows, folder dialog: the modern Explorer dialog (IFileOpenDialog with
FOS_PICKFOLDERS) through Windows PowerShell 5.1 and a small C# class
compiled by Add-Type - all part of Windows 10/11, nothing to install. If it
cannot start (Add-Type or COM blocked), the classic Shell.Application
BrowseForFolder dialog is used instead. Cancel and timeout never fall back
to the other dialog. Opening a folder: os.startfile.

Linux: zenity, then kdialog, found with shutil.which() - never installed,
never run with sudo. Opening a folder: xdg-open, then gio open. Without a
graphical session (no DISPLAY / WAYLAND_DISPLAY) nothing is attempted: the
path can always be typed into the page.

Every external program is run with an argument list (no shell). Text given
to the PowerShell scripts travels Base64-encoded, so no value can be read
as a PowerShell parameter or code, and paths come back the same way, so
non-ASCII folder names survive whatever the console code page is.
"""

import base64
import os
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

# Read at call time (not cached in the functions) so tests can simulate
# another system by assigning platform_support.PLATFORM.
PLATFORM = sys.platform

# The whole Browse request, fallback dialog included. index.html's
# browseRawDir() gives up a little earlier (85 s) on its side.
DIALOG_TIMEOUT_S = 90

# Linux: how long to wait for xdg-open/gio to report an error. Still running
# after that means the file manager is starting, which counts as success.
OPEN_FOLDER_WAIT_S = 10

NO_DIALOG_MESSAGE = (
    "No folder dialog is available on this computer. "
    "Type or paste the RAW folder path into the field."
)


class PlatformActionError(Exception):
    """Base of the errors below; the message is meant for the page."""


class ActionUnavailableError(PlatformActionError):
    """Nothing on this computer can perform the action."""


class ActionFailedError(PlatformActionError):
    """The dialog or file manager was started but failed."""


class ActionTimeoutError(PlatformActionError):
    """The dialog did not answer within DIALOG_TIMEOUT_S."""


# ======================================================================
# Public API
# ======================================================================

def choose_directory(title: str, initial_dir: str | None = None) -> str | None:
    """
    Opens the native folder dialog and returns the chosen folder as an
    absolute path, or None if the user cancelled. `initial_dir` is where the
    dialog starts when it is an existing folder (relative paths resolve
    against the current working directory, as Step 0 does); otherwise it is
    ignored. Blocks the calling thread until the dialog closes.
    """
    start = _existing_folder(initial_dir)
    if PLATFORM == "win32":
        return _choose_directory_windows(title, start)
    if PLATFORM.startswith("linux"):
        return _choose_directory_linux(title, start)
    raise ActionUnavailableError(NO_DIALOG_MESSAGE)


def open_folder(path) -> None:
    """Opens `path` in the system file manager."""
    path = Path(path)
    if PLATFORM == "win32":
        _open_folder_windows(path)
    elif PLATFORM.startswith("linux"):
        _open_folder_linux(path)
    else:
        raise ActionUnavailableError(
            f"Opening folders is not supported on this system. Open it manually: {path}"
        )


def open_browser(url: str) -> bool:
    """
    Opens url in the default web browser (standard library webbrowser:
    os.startfile on Windows, xdg-open or a known browser on Linux). Returns
    True if a browser was asked to open it, False otherwise - never raises,
    so a missing browser can never stop the caller. On Linux nothing is
    tried outside a graphical session, where webbrowser could otherwise
    start a text-mode browser in the server's own terminal.
    """
    if PLATFORM.startswith("linux") and not _graphical_session():
        return False
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:
        return False


def _existing_folder(initial_dir: str | None) -> str | None:
    if not initial_dir:
        return None
    try:
        folder = Path(initial_dir)
        return str(folder.resolve()) if folder.is_dir() else None
    except (OSError, ValueError):
        return None


def _deadline_left(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise ActionTimeoutError(_timeout_message())
    return left


def _timeout_message() -> str:
    return (
        f"The folder dialog got no answer within {DIALOG_TIMEOUT_S} s and was closed. "
        "Click Browse... again, or type the path into the field."
    )


# ======================================================================
# Windows
# ======================================================================

# Shared by both dialog scripts: Base64 (UTF-8) in, one ASCII line out -
#   OK:<Base64 path> | CANCELLED | UNAVAILABLE:<Base64 reason> | FAILED:<Base64 reason>
# UNAVAILABLE means the dialog could not be shown (-> next dialog);
# FAILED means it was shown and something went wrong afterwards.
_PS_COMMON = r"""
param([string]$TitleB64 = "", [string]$InitialDirB64 = "")
$ErrorActionPreference = "Stop"
function Decode([string]$b64) {
    if (-not $b64) { return "" }
    return [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($b64))
}
function Encode([string]$text) {
    return [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes([string]$text))
}
$title = Decode $TitleB64
$initialDir = Decode $InitialDirB64
"""

# Modern Explorer folder dialog (Common Item Dialog). Only the IFileDialog
# methods up to GetResult are declared: COM interop calls by vtable slot,
# so the order matters and the unused trailing methods can be left out.
# No owner window (IntPtr.Zero): the dialog gets its own taskbar button.
_PS_MODERN = _PS_COMMON + r"""
$source = @'
using System;
using System.Runtime.InteropServices;

public static class GnssFolderDialog
{
    [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
    private class FileOpenDialogCoClass { }

    [ComImport, Guid("d57c7288-d4ad-4768-be02-9d969532d960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileOpenDialog
    {
        [PreserveSig] int Show(IntPtr hwndOwner);
        void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
        void SetFileTypeIndex(uint iFileType);
        void GetFileTypeIndex(out uint piFileType);
        void Advise(IntPtr pfde, out uint pdwCookie);
        void Unadvise(uint dwCookie);
        void SetOptions(uint fos);
        void GetOptions(out uint pfos);
        void SetDefaultFolder(IShellItem psi);
        void SetFolder(IShellItem psi);
        void GetFolder(out IShellItem ppsi);
        void GetCurrentSelection(out IShellItem ppsi);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
        void GetResult(out IShellItem ppsi);
    }

    [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem
    {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem ppsi);
        void GetDisplayName(uint sigdnName, [MarshalAs(UnmanagedType.LPWStr)] out string ppszName);
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    private static extern int SHCreateItemFromParsingName(
        [MarshalAs(UnmanagedType.LPWStr)] string pszPath, IntPtr pbc, ref Guid riid,
        [MarshalAs(UnmanagedType.Interface)] out IShellItem ppv);

    private const uint FOS_NOCHANGEDIR = 0x00000008;
    private const uint FOS_PICKFOLDERS = 0x00000020;
    private const uint FOS_FORCEFILESYSTEM = 0x00000040;
    private const uint FOS_PATHMUSTEXIST = 0x00000800;
    private const uint SIGDN_FILESYSPATH = 0x80058000;
    private const int HRESULT_CANCELLED = unchecked((int)0x800704C7);

    public static string Pick(string title, string initialDir, out string detail)
    {
        detail = null;
        IFileOpenDialog dialog;
        try
        {
            dialog = (IFileOpenDialog)new FileOpenDialogCoClass();
            uint options;
            dialog.GetOptions(out options);
            dialog.SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);
            if (!String.IsNullOrEmpty(title)) dialog.SetTitle(title);
            if (!String.IsNullOrEmpty(initialDir))
            {
                Guid iid = typeof(IShellItem).GUID;
                IShellItem folder;
                if (SHCreateItemFromParsingName(initialDir, IntPtr.Zero, ref iid, out folder) == 0 && folder != null)
                    dialog.SetFolder(folder);
            }
        }
        catch (Exception e)
        {
            detail = e.Message;
            return "UNAVAILABLE";
        }

        int hr = dialog.Show(IntPtr.Zero);
        if (hr == HRESULT_CANCELLED) return "CANCELLED";
        if (hr != 0)
        {
            detail = String.Format("IFileOpenDialog.Show failed (HRESULT 0x{0:X8})", hr);
            return "UNAVAILABLE";
        }
        try
        {
            IShellItem item;
            dialog.GetResult(out item);
            item.GetDisplayName(SIGDN_FILESYSPATH, out detail);
            return "OK";
        }
        catch (Exception e)
        {
            detail = e.Message;
            return "FAILED";
        }
    }
}
'@
try {
    Add-Type -TypeDefinition $source -Language CSharp
} catch {
    Write-Output ("UNAVAILABLE:" + (Encode $_.Exception.Message))
    exit 0
}
$detail = $null
$status = [GnssFolderDialog]::Pick($title, $initialDir, [ref]$detail)
if ($status -eq "CANCELLED") { Write-Output "CANCELLED" } else { Write-Output ($status + ":" + (Encode $detail)) }
"""

# Classic Shell32 folder dialog (the previous default), kept as fallback.
# RootFolder stays 0 (Desktop: "This PC" and every drive): a RootFolder
# *restricts* the visible tree, so the initial folder is not used here.
_PS_LEGACY = _PS_COMMON + r"""
try {
    $shell = New-Object -ComObject Shell.Application
} catch {
    Write-Output ("UNAVAILABLE:" + (Encode $_.Exception.Message))
    exit 0
}
$BIF_RETURNONLYFSDIRS = 0x0001
$BIF_NEWDIALOGSTYLE = 0x0040
$folder = $shell.BrowseForFolder(0, $title, ($BIF_RETURNONLYFSDIRS -bor $BIF_NEWDIALOGSTYLE), 0)
if ($null -eq $folder) { Write-Output "CANCELLED" } else { Write-Output ("OK:" + (Encode $folder.Self.Path)) }
"""


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _run_powershell_dialog(script: str, title: str, start: str | None, deadline: float) -> tuple[str, str]:
    """
    Runs one dialog script in its own powershell.exe (-STA, as COM dialogs
    need) and returns (status, detail): ("OK", path), ("CANCELLED", ""),
    ("UNAVAILABLE", reason) or ("FAILED", reason). The script is written to
    a private temporary folder that is removed afterwards. stdin=DEVNULL: an
    unexpected prompt fails at once instead of hanging. A timeout kills the
    process, which closes its dialog.
    """
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        return "UNAVAILABLE", "powershell.exe not found"
    with tempfile.TemporaryDirectory(prefix="gnss_folder_dialog_") as tmp:
        script_path = Path(tmp) / "folder_dialog.ps1"
        script_path.write_text(script, encoding="ascii")
        command = [
            powershell, "-NoProfile", "-NonInteractive", "-STA",
            "-ExecutionPolicy", "Bypass", "-File", str(script_path),
            "-TitleB64", _b64(title),
        ]
        if start:
            command += ["-InitialDirB64", _b64(start)]
        try:
            result = subprocess.run(
                command, capture_output=True, timeout=_deadline_left(deadline),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ActionTimeoutError(_timeout_message()) from None
        except OSError as exc:
            return "UNAVAILABLE", f"cannot run powershell.exe: {exc}"

    stdout = result.stdout.decode("ascii", "replace")
    stderr = result.stderr.decode("utf-8", "replace").strip()
    details = f"powershell.exe exit {result.returncode}; stdout: {stdout.strip() or '(empty)'}; stderr: {stderr or '(empty)'}"
    if result.returncode != 0:
        # The script always ends with exit 0 once it runs, so the script
        # itself could not run (e.g. blocked by policy): no dialog was shown.
        return "UNAVAILABLE", details
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if last == "CANCELLED":
        return "CANCELLED", ""
    status, sep, payload = last.partition(":")
    if sep and status in ("OK", "UNAVAILABLE", "FAILED"):
        try:
            return status, base64.b64decode(payload).decode("utf-8")
        except ValueError:
            pass
    return "FAILED", details


def _choose_directory_windows(title: str, start: str | None) -> str | None:
    """
    Modern dialog first; the classic one only if the modern one could not
    be shown. A cancel or a timeout ends the request as it is.
    """
    deadline = time.monotonic() + DIALOG_TIMEOUT_S
    status, detail = _run_powershell_dialog(_PS_MODERN, title, start, deadline)
    if status == "OK":
        return detail
    if status == "CANCELLED":
        return None
    if status == "FAILED":
        raise ActionFailedError(f"The folder dialog failed: {detail}")

    modern_reason = detail
    status, detail = _run_powershell_dialog(_PS_LEGACY, title, start, deadline)
    if status == "OK":
        return detail
    if status == "CANCELLED":
        return None
    reasons = f"modern dialog: {modern_reason}; classic dialog: {detail}"
    if status == "UNAVAILABLE":
        raise ActionUnavailableError(f"{NO_DIALOG_MESSAGE} ({reasons})")
    raise ActionFailedError(f"The folder dialog failed ({reasons}).")


def _open_folder_windows(path: Path) -> None:
    startfile = getattr(os, "startfile", None)
    if startfile is None:
        raise ActionUnavailableError(f"Cannot open folders on this system. Open it manually: {path}")
    try:
        startfile(str(path))
    except OSError as exc:
        raise ActionFailedError(f"Could not open {path}: {exc}") from exc


# ======================================================================
# Linux
# ======================================================================

def _graphical_session() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _zenity_command(title: str, start: str | None) -> list[str] | None:
    zenity = shutil.which("zenity")
    if zenity is None:
        return None
    command = [zenity, "--file-selection", "--directory", f"--title={title}"]
    if start:
        # The trailing separator makes zenity open *inside* that folder.
        command.append(f"--filename={os.path.join(start, '')}")
    return command


def _kdialog_command(title: str, start: str | None) -> list[str] | None:
    kdialog = shutil.which("kdialog")
    if kdialog is None:
        return None
    return [kdialog, "--title", title, "--getexistingdirectory", start or str(Path.home())]


def _choose_directory_linux(title: str, start: str | None) -> str | None:
    """
    zenity, then kdialog: exit 0 = chosen (path on stdout), 1 = cancelled
    for both. Any other result means the dialog could not be used, so the
    next tool is tried.
    """
    if not _graphical_session():
        raise ActionUnavailableError(NO_DIALOG_MESSAGE)
    deadline = time.monotonic() + DIALOG_TIMEOUT_S
    errors = []
    for build in (_zenity_command, _kdialog_command):
        command = build(title, start)
        if command is None:
            continue
        name = Path(command[0]).name
        try:
            result = subprocess.run(
                command, capture_output=True, timeout=_deadline_left(deadline),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ActionTimeoutError(_timeout_message()) from None
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if result.returncode == 0:
            selected = os.fsdecode(result.stdout).rstrip("\r\n")
            return selected or None
        if result.returncode == 1:
            return None
        stderr = os.fsdecode(result.stderr).strip()
        errors.append(f"{name} exit {result.returncode}: {stderr or '(no message)'}")
    if errors:
        raise ActionFailedError(f"The folder dialog failed ({'; '.join(errors)}). {NO_DIALOG_MESSAGE}")
    raise ActionUnavailableError(NO_DIALOG_MESSAGE)


def _open_folder_linux(path: Path) -> None:
    """
    xdg-open, then gio open. Either returns quickly once the file manager
    has been asked to open the folder; one still running after
    OPEN_FOLDER_WAIT_S is left alone and counts as launched.
    """
    manual = f"Open it manually: {path}"
    if not _graphical_session():
        raise ActionUnavailableError(f"No graphical file manager is available on this computer. {manual}")
    candidates = [("xdg-open", []), ("gio", ["open"])]
    errors = []
    for tool, args in candidates:
        executable = shutil.which(tool)
        if executable is None:
            continue
        try:
            process = subprocess.Popen(
                [executable, *args, str(path)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            errors.append(f"{tool}: {exc}")
            continue
        try:
            _, stderr = process.communicate(timeout=OPEN_FOLDER_WAIT_S)
        except subprocess.TimeoutExpired:
            return
        if process.returncode == 0:
            return
        message = os.fsdecode(stderr or b"").strip()
        errors.append(f"{tool} exit {process.returncode}: {message or '(no message)'}")
    if errors:
        raise ActionFailedError(f"Could not open the folder ({'; '.join(errors)}). {manual}")
    raise ActionUnavailableError(
        f"No graphical file manager is available on this computer (xdg-open and gio not found). {manual}"
    )
