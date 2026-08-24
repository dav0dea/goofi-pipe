# Crash recovery

A one-minute autosave, a registry of active patches, and an offer to reopen on startup.

## The decision that shapes it

The answer here is **an autosave plus a registry**, not `read_stable`, not an fsync sequence, and
not a journal. The workspace mount is disposable and a `.gfi` is written whole, so the risk is not a
torn file — it is an hour of unsaved authoring.

## The parts

- **Autosave.** Every minute, and only when the patch is dirty. It writes beside the save path, or
  into the config folder for a patch that has never been saved.
- **A flocked registry** of the patches an instance holds open. The lock is what separates "this
  patch is open in another window" from "this patch died with work in it".
- **An offer on startup**, never an automatic restore. The user decides whether the autosave or the
  file on disk is the real one, and must be able to see which is which.

## Needs

- What an autosave does to `unsaved_changes`. Nothing — it is not a save, and it must not clear the
  dot.
- Where an autosave for an unsaved patch lives, and when it is swept.
- The recovery UI, which is the only part with no precedent in the app today.
