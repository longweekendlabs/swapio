# Changelog

## 0.4.5 - 2026-09-03

- Keep the destination photo's own eyeballs, moved onto the swapped eyelids so they line up.
- Fix the hard white catchlight and flat iris the 256px swapper draws in place of a real eye.
- Add a restoration strength control and lower the default, so Best sharpens without repainting the face.
- Default new installations to JPEG output instead of PNG.

## 0.4.4 - 2026-09-03

- Fix every model download failing certificate verification on Fedora and other non-Debian systems.
- Verify downloads against the certificate bundle inside the package instead of the build machine's OpenSSL paths.

## 0.4.3 - 2026-09-02

- Add a **Best** processing quality that restores the swapped face with GPEN-BFR 1024.
- Recover eyelash, eyelid, iris and tooth detail that the 256px swapper cannot resolve.
- Fix soft swaps caused by bilinear resampling: align and paste faces with Lanczos instead.
- Download and verify the face restoration model alongside the existing four.
- Bump the batch-processing revision so existing photos are regenerated with the sharper resampling.

## 0.4.2 - 2026-08-06

- Detect close-up faces and raise Careful processing from 512px to 768px or 1024px automatically.
- Preserve more of the target teeth edges and use scale-aware mouth feathering on large faces.
- Enable inner-mouth and teeth preservation by default for new installations.
- Remove hair color, face/body skin-tone matching, appearance-model downloads, and their runtime.
- Keep Model Management focused on the detector, identity encoder, and two face swappers.
- Bump the batch-processing revision so existing photos are regenerated with the close-up fix.

## 0.4.1 - 2026-08-06

- Add Settings and Model Management pages under the hamburger menu.
- Add a resumable, checksum-verified optional FASHN Human Parser download.
- Extend source skin-tone matching to visible torso, arms, hands, legs, and feet.
- Keep target lighting and texture while transferring source tone in preview and batch.
- Pause hair recoloring after the evaluated large model failed real-photo quality checks.
- Keep optional appearance models out of the application package and Git history.

## 0.4.0 - 2026-08-06

- Add offline photo appearance controls to preview and batch processing.
- Add natural hair-color presets, a native custom color picker, and source-hair matching.
- Add a hair-color strength control that retains local texture, highlights, and shadows.
- Add conservative source skin-tone matching for the selected face, ears, and neck.
- Use a verified MIT-licensed BiSeNet ONNX parser to isolate appearance regions.
- Reject disconnected hair belonging to a nearby person instead of recoloring it.
- Include every appearance setting in repeat-batch history so changed settings rerun.
- Keep the source, destination, and preview canvases aligned above equal-height footers.
- Make inner-mouth preservation opt-in by default after real-photo quality evaluation.

## 0.3.1 - 2026-08-06

- Add an enabled-by-default option to preserve the target's inner mouth and teeth.
- Use the Buffalo 106-point landmark model to isolate the inner-lip contour precisely.
- Keep lips and surrounding facial identity swapped while restoring only teeth, tongue,
  and oral-cavity pixels with a softly feathered mask.
- Avoid applying the preservation mask when the lips are closed.
- Include the mouth-preservation setting in repeat-batch history so changed settings rerun.
- Keep the source and preview canvases inside their cards at compact window heights.

## 0.3.0 - 2026-08-06

- Recalculate the safe sibling output folder whenever a destination folder is selected.
- Keep a persistent `.swapio-history.json` manifest inside each output folder.
- Skip unchanged photos already completed with the same source and processing settings.
- Reprocess photos when an input changes, settings change, the output is removed, or the
  repeat-batch protection is deliberately disabled.
- Report newly saved, unchanged, and failed photos as separate batch counts.

## 0.2.0 - 2026-08-05

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

## 0.1.0 - 2026-08-05

- Initial offline still-image preview and batch swapping workflow.
