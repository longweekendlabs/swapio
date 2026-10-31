"""Run the computer's media tools outside any private application libraries."""
import os
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
