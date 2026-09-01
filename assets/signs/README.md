# Rust paintable icons

These thumbnails identify Rust's built-in paintable items in the premade
profile picker. They are resized copies of the corresponding in-game item
icons. `tools/download_sign_icons.py` refreshes them from Facepunch's item CDN,
with RustLabs as a fallback for current DLC icons whose old CDN paths return
404.

The filenames are Rust `ItemDefinition.shortname` values so the catalog and
artwork cannot drift apart silently.
