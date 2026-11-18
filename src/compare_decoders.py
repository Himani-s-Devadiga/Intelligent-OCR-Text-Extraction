
# ==========================================
# GREEDY VS BEAM SEARCH COMPARISON
# ==========================================

characters = ["_", "A", "B", "C", "T"]
blank_index = 0

# Sample model predictions
predictions = [
    [0.10, 0.60, 0.20, 0.05, 0.05],
    [0.10, 0.20, 0.60, 0.05, 0.05],
    [0.10, 0.70, 0.10, 0.05, 0.05]
]


# ==========================================
# 1. GREEDY DECODING
# ==========================================

def greedy_decoder(predictions):

    best_path = []

    for probabilities in predictions:

        best_index = probabilities.index(max(probabilities))

        best_path.append(best_index)

    decoded = []
    previous = None

    for index in best_path:

        character = characters[index]

        if character != previous and character != "_":
            decoded.append(character)

        previous = character

    return "".join(decoded)


# ==========================================
# 2. CTC PREFIX BEAM SEARCH
# ==========================================

def ctc_beam_search(predictions, beam_width=3):

    beams = {"": (1.0, 0.0)}

    for probabilities in predictions:

        new_beams = {}

        def update(prefix, pb=0.0, pnb=0.0):

            old_pb, old_pnb = new_beams.get(
                prefix, (0.0, 0.0)
            )

            new_beams[prefix] = (
                old_pb + pb,
                old_pnb + pnb
            )

        for prefix, (pb, pnb) in beams.items():

            total = pb + pnb

            # Blank extension
            update(
                prefix,
                pb=total * probabilities[blank_index]
            )

            # Character extensions
            for index, character in enumerate(characters):

                if index == blank_index:
                    continue

                probability = probabilities[index]

                if probability == 0:
                    continue

                if prefix and character == prefix[-1]:

                    # Repeated character without blank
                    update(
                        prefix,
                        pnb=pnb * probability
                    )

                    # Repeated character after blank
                    update(
                        prefix + character,
                        pnb=pb * probability
                    )

                else:

                    update(
                        prefix + character,
                        pnb=total * probability
                    )

        sorted_beams = sorted(
            new_beams.items(),
            key=lambda item: sum(item[1]),
            reverse=True
        )

        beams = dict(sorted_beams[:beam_width])

    return max(
        beams,
        key=lambda prefix: sum(beams[prefix])
    )


# ==========================================
# 3. COMPARE RESULTS
# ==========================================

greedy_result = greedy_decoder(predictions)

beam_result = ctc_beam_search(
    predictions,
    beam_width=3
)

print("GREEDY DECODING:", greedy_result)

print("BEAM SEARCH:", beam_result)

print("\nComparison Complete!")
