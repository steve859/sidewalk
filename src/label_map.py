VIOLATION_TYPES = [
    "parked_motorbike",
    "parked_car",
    "street_vendor",
    "construction_material",
    "advertisement_board",
    "garbage_bag",
    "temporary_construction",
]

VIOLATION_TO_IDX = {
    label: idx for idx, label in enumerate(VIOLATION_TYPES)
}

IDX_TO_VIOLATION = {
    idx: label for label, idx in VIOLATION_TO_IDX.items()
}

NUM_CLASSES = len(VIOLATION_TYPES)