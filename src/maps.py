"""
src/maps.py
Contains custom grid layouts for FrozenLake.
"""

MAP_LAYOUTS = {
    # Simple test maps (no holes - verify agent works)
    "2x2_simple": [
        "SF",
        "FG"
    ],
    "3x3_simple": [
        "SFF",
        "FFF",
        "FFG"
    ],
    # Maps with 1 hole (11% density for 3x3, 6% for 4x4)
    "3x3_1hole": [  
        "SFF",
        "FHF",
        "FFG"
    ],
    "4x4_1hole": [
        "SFFF",
        "FHFF",
        "FFFF",
        "FFFG"
    ],
    # Maps with appropriate hole density (10-20%)
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "6x6": [
        "SFFFFF",
        "FFFHFF",
        "FHHFFF",
        "FFFFHF",
        "FFFHFF",
        "HFFFFG"
    ]
}

def get_map(size_input):
    """
    Retrieves the map layout by name or by integer size.
    
    Examples:
        get_map(2)     -> returns 2x2 map
        get_map("4x4") -> returns 4x4 map
    """
    if isinstance(size_input, int):
        map_key = f"{size_input}x{size_input}"
    else:
        map_key = str(size_input)

    if map_key in MAP_LAYOUTS:
        return MAP_LAYOUTS[map_key]
    else:
        raise ValueError(f"Map '{map_key}' is not defined in src/maps.py. Available maps: {list(MAP_LAYOUTS.keys())}")