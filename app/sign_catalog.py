"""Known Rust paintable surfaces, read from the game's prefab assets.

The dimensions in this module mirror ``tools/sign_sizes.json``.  They are
kept in application code so a packaged build can plan a faithful preview
without reading a player's multi-gigabyte Rust bundles at runtime.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SignCatalogEntry:
    id: str
    name: str
    category: str
    width: int
    height: int
    item_shortname: str
    aliases: tuple[str, ...] = ()

    @property
    def texture_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def search_text(self) -> str:
        return " ".join((self.name, self.category, self.item_shortname, *self.aliases))


def _entry(
    id: str,
    name: str,
    category: str,
    size: tuple[int, int],
    shortname: str,
    *aliases: str,
) -> SignCatalogEntry:
    return SignCatalogEntry(id, name, category, size[0], size[1], shortname, aliases)


# Friendly item names come from Rust's ItemDefinition records. Texture sizes
# come from each prefab's MeshPaintableSource.texWidth/texHeight fields.
SIGN_CATALOG: tuple[SignCatalogEntry, ...] = (
    _entry("artist-small", "Artist Canvas Small", "Artist Canvases", (192, 256), "sign.artistcanvas.xs", "xs canvas"),
    _entry("artist-medium", "Artist Canvas Medium", "Artist Canvases", (320, 240), "sign.artistcanvas.s", "s canvas"),
    _entry("artist-large", "Artist Canvas Large", "Artist Canvases", (320, 240), "sign.artistcanvas.m", "m canvas"),
    _entry("artist-standing", "Artist Canvas Standing", "Artist Canvases", (256, 640), "sign.artistcanvas.l", "l canvas", "tall canvas"),
    _entry("artist-xl", "Artist Canvas XL", "Artist Canvases", (512, 512), "sign.artistcanvas.xl", "extra large canvas"),
    _entry("artist-xxl", "Artist Canvas XXL", "Artist Canvases", (1024, 512), "sign.artistcanvas.xxl", "extra extra large canvas"),
    _entry("lightup-small", "Light-Up Frame Small", "Light-Up Frames", (128, 175), "lightupframe.small", "light frame s"),
    _entry("lightup-medium", "Light-Up Frame Medium", "Light-Up Frames", (320, 240), "lightupframe.medium", "light frame m"),
    _entry("lightup-large", "Light-Up Frame Large", "Light-Up Frames", (320, 256), "lightup.large", "light frame l"),
    _entry("lightup-standing", "Light-Up Frame Standing", "Light-Up Frames", (128, 320), "lightupframe.standing", "light frame tall"),
    _entry("lightup-xl", "Light-Up Frame XL", "Light-Up Frames", (512, 512), "lightup.xl", "light frame extra large"),
    _entry("lightup-xxl", "Light-Up Frame XXL", "Light-Up Frames", (1024, 512), "lightup.xxl", "xxl light frame"),
    _entry("ornate-small", "Ornate Frame Small", "Ornate Frames", (128, 175), "goldframe.small", "gold frame s"),
    _entry("ornate-medium", "Ornate Frame Medium", "Ornate Frames", (320, 240), "goldframe.medium", "gold frame m"),
    _entry("ornate-large", "Ornate Frame Large", "Ornate Frames", (320, 256), "goldframe.large", "gold frame l"),
    _entry("ornate-standing", "Ornate Frame Standing", "Ornate Frames", (128, 320), "goldframe.standing", "gold frame tall"),
    _entry("ornate-xl", "Ornate Frame XL", "Ornate Frames", (512, 512), "goldframe.xl", "gold frame extra large"),
    _entry("ornate-xxl", "Ornate Frame XXL", "Ornate Frames", (1024, 512), "goldframe.xxl", "gold frame extra extra large"),
    _entry("shutter-small", "Shutter Frame Small", "Shutter Frames", (128, 175), "scrapframe.small", "scrap frame s"),
    _entry("shutter-medium", "Shutter Frame Medium", "Shutter Frames", (320, 240), "scrapframe.medium", "scrap frame m"),
    _entry("shutter-large", "Shutter Frame Large", "Shutter Frames", (320, 256), "scrapframe.large", "scrap frame l"),
    _entry("shutter-standing", "Shutter Frame Standing", "Shutter Frames", (128, 320), "scrapframe.standing", "scrap frame tall"),
    _entry("shutter-xl", "Shutter Frame XL", "Shutter Frames", (512, 512), "scrapframe.xl", "scrap frame extra large"),
    _entry("shutter-xxl", "Shutter Frame XXL", "Shutter Frames", (1024, 512), "scrapframe.xxl", "scrap frame extra extra large"),
    _entry("picture-landscape", "Landscape Picture Frame", "Picture Frames", (256, 192), "sign.pictureframe.landscape", "horizontal picture"),
    _entry("picture-portrait", "Portrait Picture Frame", "Picture Frames", (205, 256), "sign.pictureframe.portrait", "vertical picture"),
    _entry("picture-tall", "Tall Picture Frame", "Picture Frames", (128, 512), "sign.pictureframe.tall"),
    _entry("picture-xl", "XL Picture Frame", "Picture Frames", (512, 512), "sign.pictureframe.xl"),
    _entry("picture-xxl", "XXL Picture Frame", "Picture Frames", (1024, 512), "sign.pictureframe.xxl"),
    _entry("wood-small", "Small Wooden Sign", "Wooden Signs", (256, 128), "sign.wooden.small"),
    _entry("wood-medium", "Medium Wooden Sign", "Wooden Signs", (512, 256), "sign.wooden.medium"),
    _entry("wood-large", "Large Wooden Sign", "Wooden Signs", (512, 256), "sign.wooden.large"),
    _entry("wood-huge", "Huge Wooden Sign", "Wooden Signs", (1024, 256), "sign.wooden.huge"),
    _entry("hanging", "Two Sided Hanging Sign", "Hanging Signs", (512, 256), "sign.hanging", "horizontal hanging"),
    _entry("hanging-ornate", "Two Sided Ornate Hanging Sign", "Hanging Signs", (320, 384), "sign.hanging.ornate"),
    _entry("hanging-banner", "Large Banner Hanging", "Banners", (256, 1024), "sign.hanging.banner.large"),
    _entry("pole-banner", "Large Banner on Pole", "Banners", (256, 1024), "sign.pole.banner.large"),
    _entry("neon-small", "Small Neon Sign", "Neon Signs", (128, 128), "sign.neon.125x125"),
    _entry("neon-medium", "Medium Neon Sign", "Neon Signs", (256, 128), "sign.neon.125x215"),
    _entry("neon-medium-animated", "Medium Animated Neon Sign", "Neon Signs", (256, 128), "sign.neon.125x215.animated"),
    _entry("neon-large", "Large Neon Sign", "Neon Signs", (256, 256), "sign.neon.xl"),
    _entry("neon-large-animated", "Large Animated Neon Sign", "Neon Signs", (256, 256), "sign.neon.xl.animated"),
    _entry("post-single", "Single Sign Post", "Sign Posts", (256, 128), "sign.post.single"),
    _entry("post-double", "Double Sign Post", "Sign Posts", (512, 512), "sign.post.double"),
    _entry("post-town-single", "One Sided Town Sign Post", "Sign Posts", (512, 256), "sign.post.town.roof"),
    _entry("post-town-double", "Two Sided Town Sign Post", "Sign Posts", (512, 256), "sign.post.town"),
    _entry("paintable-window", "Paintable Window", "Other Paintables", (512, 256), "window.paintable"),
    _entry("spinner-wheel", "Spinning Wheel", "Other Paintables", (512, 512), "spinner.wheel"),
)


def catalog_entry(entry_id: str) -> SignCatalogEntry | None:
    return next((entry for entry in SIGN_CATALOG if entry.id == entry_id), None)


def _words(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def fuzzy_score(query: str, entry: SignCatalogEntry) -> float:
    """A forgiving search score supporting typos, initials and word order."""

    query_words = _words(query)
    if not query_words:
        return 1.0
    candidate_words = _words(entry.search_text)
    name_words = _words(entry.name)
    candidate = " ".join(candidate_words)
    compact_query = "".join(query_words)
    compact_candidate = "".join(candidate_words)
    if all(word in name_words for word in query_words):
        coverage = 1.05
    elif all(word in candidate_words for word in query_words):
        coverage = 1.0
    elif all(any(word in candidate_word for candidate_word in candidate_words) for word in query_words):
        coverage = 0.92
    else:
        coverage = sum(
            max((SequenceMatcher(None, word, candidate_word).ratio() for candidate_word in candidate_words), default=0.0)
            for word in query_words
        ) / len(query_words)
    sequence = SequenceMatcher(None, compact_query, compact_candidate).ratio()
    initials = "".join(word[0] for word in candidate_words if word)
    initial_score = SequenceMatcher(None, compact_query, initials).ratio()
    return max(coverage, sequence, initial_score * 0.9)


def search_catalog(query: str) -> list[SignCatalogEntry]:
    scored = [(fuzzy_score(query, entry), entry) for entry in SIGN_CATALOG]
    if query.strip():
        scored = [pair for pair in scored if pair[0] >= 0.48]
    return [entry for _score, entry in sorted(scored, key=lambda pair: (-pair[0], pair[1].category, pair[1].name))]


@lru_cache(maxsize=None)
def catalog_asset_root() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "assets" / "signs"
    return Path(__file__).resolve().parent.parent / "assets" / "signs"


def catalog_icon_path(entry: SignCatalogEntry) -> Path:
    return catalog_asset_root() / f"{entry.item_shortname}.png"


__all__ = [
    "SIGN_CATALOG",
    "SignCatalogEntry",
    "catalog_entry",
    "catalog_icon_path",
    "fuzzy_score",
    "search_catalog",
]
