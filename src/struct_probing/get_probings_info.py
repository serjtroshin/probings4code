from src.struct_probing.probings import supported_probings

probings_info = {}
for probing in supported_probings:
    p = supported_probings[probing]
    probings_info[probing] = (
        p.get_augmentation(),
        p.get_dataset(),
        p.get_embedding_type(),
    )
print(str(probings_info))
