"""Deterministic ordinal choice and multi-tile response checks for F/G."""


def sorted_candidates(boxes):
    """Use the same left-to-right order as the rendered puzzle."""
    return sorted(boxes, key=lambda box: (box[0] + box[2]) / 2)


def g_choice(boxes, rank, direction):
    """Return (tile index, original unpadded box); None means puzzle failure."""
    ordered = sorted_candidates(boxes)
    if not isinstance(rank, int) or rank < 1 or rank > len(ordered):
        return None, None
    if direction == 'left_to_right':
        index = rank - 1
    elif direction == 'right_to_left':
        index = len(ordered) - rank
    else:
        raise ValueError(f'Unsupported direction: {direction}')
    return index, ordered[index]


def response_spans_multiple_tiles(boxes, spans, min_tile_overlap=0.25):
    """True when a response gives multiple boxes or one box covers multiple tiles."""
    if len(boxes) > 1:
        return True
    if not boxes:
        return False
    left, right = boxes[0][0], boxes[0][2]
    touched = sum(
        max(0.0, min(right, tile_right) - max(left, tile_left))
        / (tile_right - tile_left) >= min_tile_overlap
        for tile_left, tile_right in spans
        if tile_right > tile_left
    )
    return touched > 1
