
# Character vocabulary for CTC decoding

characters = ["_", "A", "B", "C", "T"]

# Blank token index
blank_index = 0

print("Character vocabulary:", characters)
print("Blank token:", characters[blank_index])


def greedy_decoder(predictions):

    decoded_text = []
    previous_character = None

    for index in predictions:

        character = characters[index]

        # Remove consecutive repeated characters
        if character != previous_character:

            # Ignore blank tokens
            if character != "_":
                decoded_text.append(character)

        previous_character = character

    return "".join(decoded_text)


# Sample model predictions
predictions = [3, 3, 0, 1, 1, 4, 4]

# Decode predictions
result = greedy_decoder(predictions)

print("Raw predictions:", predictions)
print("Decoded text:", result)


# Test repeated characters separated by blank

test_predictions = [3, 0, 3]

test_result = greedy_decoder(test_predictions)

print("\nRepeated character test:")
print("Raw predictions:", test_predictions)
print("Decoded text:", test_result)
predictions = [3, 0, 3]