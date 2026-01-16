def get_map(size=6):
    """
    Returns a custom map based on size.
    S = Start, G = Goal, F = Frozen (Safe), H = Hole
    """
    if size ==4:
        # 4x4 Map 
        return [
            "SFHH",
            "FFFH",
            "FFFF",
            "FFFG"
        ]
    if size == 6:
        # 6x6 Map (~14% Holes)
    
        return [
            "SFHFHH",
            "FFFFFH",
            "FFFFFF",
            "FFFFHF",
            "FFFFFF",
            "HFFFFG"
        ]
    elif size == 8:
        # 8x8 Map (~15% Holes)
        return [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFHFFFFF",
            "FFHHFFFF",
            "FFFFFHFF",
            "FHFFFFFF",
            "FFFFHFFF",
            "HFFFFFFG"
        ]
    else:
        raise ValueError("Only 6x6 and 8x8 maps are defined.")