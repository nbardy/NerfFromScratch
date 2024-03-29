# Holds some geometric constants
face_directions = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
]
corner_directions = [
    (1, 1, 1),
    (-1, -1, -1),
    (1, -1, -1),
    (-1, 1, 1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, 1, -1),
]
temporal_directions = [
    (0, 0, 0, 1),
    (0, 0, 0, -1),
]  # For time only, space remains unchanged

# Sampling corner neighbors in time direction exponentially
exponential_temporal_directions = [
    (0, 0, 0, -64),
    (0, 0, 0, -8),
    (0, 0, 0, -4),
    (0, 0, 0, -2),
    (0, 0, 0, -1),
    (0, 0, 0, 0),
    (0, 0, 0, 1),
    (0, 0, 0, 2),
    (0, 0, 0, 4),
    (0, 0, 0, 8),
    (0, 0, 0, 64),
]
