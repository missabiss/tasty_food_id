import tensorflow as tf
import pandas as pd 
from keras.preprocessing.sequence import pad_sequences
import string

ingr_map = pd.read_csv("../data/ingr_map_mappings.csv")
ingr_dict = {}
for index, row in ingr_map.iterrows():
    ingr_dict[row["replaced"]] = row["id"]

alphabet = list(string.ascii_lowercase)
alphabet.insert(0, ' ')

def load_nlp_model(path="models/best_nlp.keras"):
    return tf.keras.models.load_model(path)


def evaluate_ingredients(model, raw_ingredients):
    """
    model: trained food rating model
    raw_ingredients: list of strings
    prints food rating prediction
    """
    processed = process_ingredients(raw_ingredients)
    rating = model.predict(processed[1]).argmax(axis=1)
    rating_map = {5: 'Delicious!',
                  4: 'Pretty good!',
                  3: 'Okay',
                  2: 'Questionable',
                  1: 'That will be bad.'}
    print(f"I predict that {processed[0]} will have a flavor level of: {rating[0]}, {rating_map[rating[0]]}")


def process_ingredients(ingredients: list):
    """
    ingredients: list of ingredient (singular) strings
    returns: tuple of successfully processed ingredients and encoded, padded ingredient list
    """
    encoded = []
    kept = []
    lost = []
    for ingr in ingredients:
        if ingr_dict.get(ingr):
            encoded.append(ingr_dict.get(ingr))
            kept.append(ingr)
        else:
            lost.append(ingr)
    if len(lost) > 0:
        print("Sorry, I could not find codes for: ", lost)
        print(f"But I can predict for {kept}")
    else:
        print(f'Predicting for {kept}')
    return kept, pad_sequences(
        pd.Series([encoded]), maxlen=100, padding="pre", truncating="pre", value=0
    )

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(alphabet), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits = self.model(inputs=input_ids)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars

def generate_text(model, input):
    """
    model: trained food text model
    input: a string
    prints predicted text
    """
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    next_char = tf.constant([input])
    result = []

    for n in range(100):
        next_char = one_step_model.generate_one_step(next_char)
        result.append(next_char)

    result = tf.strings.join(result)
    print(f'I think it will taste like...')
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)