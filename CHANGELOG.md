# Changelog

## 0.2.0 — 2026-08-05

- Prevent black HyperSwap previews caused by non-finite CUDA model output.
- Keep CUDA face detection while using the stable CPU path for quality swaps.
- Reject corrupted model pixels before they can be pasted or saved.
- Automatically propose a safe sibling output folder and enable batch swapping without an extra browse step.
- Make 512px HyperSwap processing the default quality mode.
- Add balanced 256px HyperSwap and fast 128px InSwapper modes.
- Save lossless PNG by default so pixels outside the face composite remain exact.
- Add an optional JPEG 98 output mode for smaller files.
- Use native KDE/Zenity file dialogs and remember each picker location.
- Show the version and Long Weekend Labs credit in the main window.
- Add a hamburger menu with About, GitHub, and Quit actions.
- Add a first-run model setup dialog with license acknowledgement, live download progress,
  verification status, cancellation, retry, and offline-ready confirmation.

## 0.1.0 — 2026-08-05

- Initial offline still-image preview and batch swapping workflow.
