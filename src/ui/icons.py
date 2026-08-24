"""Flat, self-contained SVG icons (stroke-based, currentColor) used in place of emoji.

Kept as raw inline SVG so cards and headers render consistently across
platforms/fonts, instead of relying on emoji glyphs that vary by OS.
"""

_TEMPLATE = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' "
    "viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
    "stroke-linecap='round' stroke-linejoin='round' style='vertical-align:-4px;'>{body}</svg>"
)

_PATHS = {
    "hospital": (
        "<path d='M4 21V6a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v15'/>"
        "<path d='M14 21v-9a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v9'/>"
        "<path d='M9 21v-4'/>"
        "<path d='M7 9h4'/><path d='M9 7v4'/>"
        "<path d='M3 21h18'/>"
    ),
    "stethoscope": (
        "<path d='M5 4v6a4 4 0 0 0 8 0V4'/>"
        "<path d='M5 4H4'/><path d='M9 4H8'/>"
        "<path d='M13 10v2a6 6 0 0 0 12 0v-1'/>"
        "<circle cx='19' cy='6' r='2'/>"
        "<circle cx='9' cy='18' r='3'/>"
    ),
    "chart-bar": (
        "<path d='M3 3v18h18'/>"
        "<rect x='7' y='12' width='3' height='6'/>"
        "<rect x='12' y='8' width='3' height='10'/>"
        "<rect x='17' y='5' width='3' height='13'/>"
    ),
    "search": (
        "<circle cx='11' cy='11' r='7'/>"
        "<path d='m21 21-4.3-4.3'/>"
    ),
    "info": (
        "<circle cx='12' cy='12' r='9'/>"
        "<path d='M12 11v6'/>"
        "<path d='M12 7.5v.01'/>"
    ),
    "arrow-left": (
        "<path d='M19 12H5'/>"
        "<path d='m11 18-6-6 6-6'/>"
    ),
    "check": (
        "<path d='M20 6 9 17l-5-5'/>"
    ),
    "alert-triangle": (
        "<path d='M12 9v4'/>"
        "<path d='M12 17h.01'/>"
        "<path d='m10.29 3.86-8.16 14.14A1.5 1.5 0 0 0 3.5 20.5h17a1.5 1.5 0 0 0 1.37-2.5L13.71 3.86a1.5 1.5 0 0 0-2.42 0Z'/>"
    ),
    "rotate-ccw": (
        "<path d='M3 12a9 9 0 1 0 3-6.7L3 8'/>"
        "<path d='M3 3v5h5'/>"
    ),
    "history": (
        "<path d='M3 12a9 9 0 1 0 3-6.7L3 8'/>"
        "<path d='M3 3v5h5'/>"
        "<path d='M12 7v5l4 2'/>"
    ),
    "users": (
        "<path d='M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2'/>"
        "<circle cx='9' cy='7' r='4'/>"
        "<path d='M23 21v-2a4 4 0 0 0-3-3.87'/>"
        "<path d='M16 3.13a4 4 0 0 1 0 7.75'/>"
    ),
    "user": (
        "<circle cx='12' cy='8' r='4'/>"
        "<path d='M4 21v-1a8 8 0 0 1 16 0v1'/>"
    ),
    "calendar": (
        "<rect x='3' y='4' width='18' height='18' rx='2'/>"
        "<path d='M16 2v4'/><path d='M8 2v4'/><path d='M3 10h18'/>"
    ),
}


def icon(name: str, size: int = 20) -> str:
    """Return an inline SVG string for the given icon name."""
    return _TEMPLATE.format(size=size, body=_PATHS[name])
