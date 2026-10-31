"""Run the computer's media tools outside any private application libraries."""
import os
import shutil
import subprocess
import sys


def native_process_environment() -> dict[str, str]:
    environment = os.environ.copy()
    original = environment.get("LD_LIBRARY_PATH_ORIG")
    if original:
        environment["LD_LIBRARY_PATH"] = original
    elif getattr(sys, "frozen", False):
        environment.pop("LD_LIBRARY_PATH", None)
    if getattr(sys, "frozen", False):
        for variable in ("QT_PLUGIN_PATH", "QML2_IMPORT_PATH", "QML_IMPORT_PATH",
                         "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            original = environment.get(f"{variable}_ORIG")
            if original is not None:
                environment[variable] = original
            else:
                environment.pop(variable, None)
    return environment


def open_web_url(url: str) -> bool:
    """Hand a web link to the system browser outside bundled app libraries."""
    if not url.startswith(('https://', 'http://')):
        return False
    if sys.platform == 'win32':
        try:
            os.startfile(url)
            return True
        except OSError:
            return False
    candidates = [('open', [url])] if sys.platform == 'darwin' else [
        ('kioclient6', ['exec', url]), ('kioclient5', ['exec', url]), ('xdg-open', [url])]
    for program, arguments in candidates:
        executable = shutil.which(program)
        if executable:
            try:
                subprocess.Popen([executable, *arguments], env=native_process_environment(),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                return True
            except OSError:
                continue
    return False
