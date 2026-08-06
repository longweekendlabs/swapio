# Changelog

## 0.3.1 — 2026-08-06

- Add an enabled-by-default option to preserve the target's inner mouth and teeth.
- Use the Buffalo 106-point landmark model to isolate the inner-lip contour precisely.
- Keep lips and surrounding facial identity swapped while restoring only teeth, tongue,
  and oral-cavity pixels with a softly feathered mask.
- Avoid applying the preservation mask when the lips are closed.
- Include the mouth-preservation setting in repeat-batch history so changed settings rerun.
- Keep the source and preview canvases inside their cards at compact window heights.

## 0.3.0 — 2026-08-06

- Recalculate the safe sibling output folder whenever a destination folder is selected.
- Keep a persistent `.swapio-history.json` manifest inside each output folder.
- Skip unchanged photos already completed with the same source and processing settings.
- Reprocess photos when an input changes, settings change, the output is removed, or the
  repeat-batch protection is deliberately disabled.
- Report newly saved, unchanged, and failed photos as separate batch counts.

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
- Fix native KDE file and folder pickers in packaged builds by isolating system Qt libraries
  from Swapio's bundled Qt runtime, with a fallback if a native picker crashes.
- Restore CUDA support in a model-less public RPM profile and report the provider used by
  loaded model sessions.
- Keep the CUDA RPM portable and below the release-size target by using system CUDA 12/cuDNN 9
  libraries rather than copying the build machine's complete toolkit.
- Open completed output folders through the native file manager with an isolated environment.
- Keep destination action buttons uniform and readable at the minimum window size.
- Name outputs as `original_swapped_DDMMYYYY-HHMMSS.ext`, with collision protection.
- Allow each source portrait to have a remembered character name, producing
  `CharacterName_swapped_DDMMYYYY-HHMMSS.ext` instead of camera-origin filenames.

## 0.1.0 — 2026-08-05

- Initial offline still-image preview and batch swapping workflow.
