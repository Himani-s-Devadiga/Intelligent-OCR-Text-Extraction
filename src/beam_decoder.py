
# ==========================================
# CTC PREFIX BEAM SEARCH DECODER
# ==========================================

# Character vocabulary
characters = ["_", "A", "B", "C", "T"]

# Blank token index
blank_index = 0


# ==========================================
# 1. PROBABILITY ADDITION
# ==========================================

def update_beam(beams, prefix, pb=0.0, pnb=0.0):

    old_pb, old_pnb = beams.get(prefix, (0.0, 0.0))

    beams[prefix] = (
        old_pb + pb,
        old_pnb + pnb
    )


# ==========================================
# 2. CTC PREFIX BEAM SEARCH
# ==========================================

def ctc_beam_search(predictions, beam_width=3):

    # Initially, no characters have been recognized
    beams = {
        "": (1.0, 0.0)
    }

    # Process each time step
    for timestep, probabilities in enumerate(predictions):

        new_beams = {}

        # Process every existing prefix
        for prefix, (pb, pnb) in beams.items():

            total_probability = pb + pnb

            # ----------------------------------
            # CASE 1: BLANK TOKEN
            # ----------------------------------

            blank_probability = probabilities[blank_index]

            new_pb = total_probability * blank_probability

            update_beam(
                new_beams,
                prefix,
                pb=new_pb
            )

            # ----------------------------------
            # CASE 2: CHARACTER TOKENS
            # ----------------------------------

            for index, character in enumerate(characters):

                if index == blank_index:
                    continue

                char_probability = probabilities[index]

                if char_probability == 0:
                    continue

                # Repeated character
                if prefix and character == prefix[-1]:

                    # Same prefix: repeated character without blank
                    update_beam(
                        new_beams,
                        prefix,
                        pnb=pnb * char_probability
                    )

                    # New repeated character after blank
                    extended_prefix = prefix + character

                    update_beam(
                        new_beams,
                        extended_prefix,
                        pnb=pb * char_probability
                    )

                else:

                    # Add a new character
                    extended_prefix = prefix + character

                    update_beam(
                        new_beams,
                        extended_prefix,
                        pnb=total_probability * char_probability
                    )

        # ----------------------------------
        # CASE 3: BEAM PRUNING
        # ----------------------------------

        sorted_beams = sorted(
            new_beams.items(),
            key=lambda item: item[1][0] + item[1][1],
            reverse=True
        )

        beams = dict(sorted_beams[:beam_width])

        print(f"\nTime Step {timestep + 1}")

        for prefix, (pb, pnb) in beams.items():

            print(
                repr(prefix),
                "Total:",
                round(pb + pnb, 4),
                "Blank:",
                round(pb, 4),
                "Non-blank:",
                round(pnb, 4)
            )

    # --------------------------------------
    # 4. FINAL DECODING
    # --------------------------------------

    best_prefix = max(
        beams,
        key=lambda prefix: sum(beams[prefix])
    )

    return best_prefix


# ==========================================
# 3. SAMPLE MODEL PREDICTIONS
# ==========================================

predictions = [

    [0.10, 0.80, 0.10, 0.00, 0.00],

    [0.70, 0.10, 0.20, 0.00, 0.00],

    [0.10, 0.80, 0.10, 0.00, 0.00]

]


# ==========================================
# 4. RUN THE DECODER
# ==========================================

print("Character Vocabulary:", characters)

print("Beam Width: 3")

result = ctc_beam_search(
    predictions,
    beam_width=3
)

print("\nFinal Recognized Text:", result)
