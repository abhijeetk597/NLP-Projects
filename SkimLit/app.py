from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from spacy.lang.en import English

app = FastAPI()

nlp = English()
sentencizer = nlp.add_pipe("sentencizer")

one_hot = joblib.load("SkimLit\one_hot_encoder_skimlit.joblib", mmap_mode=None)

model = tf.keras.models.load_model('Skimlit\skimlit.h5',
                                    custom_objects={'KerasLayer': hub.KerasLayer})

class RctAbstract(BaseModel):
    abstract: str

def split_chars(text):
    return " ".join(list(text))

def transform_abstract(abstract):
    doc = nlp(abstract)
    abstract_lines = [str(sent) for sent in list(doc.setns)]
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
    # Get total number of lines
    total_lines_in_sample = len(abstract_lines)

    # Go through each line in abstract and create a list of dictionaries containing features
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict["text"] = str(line)
        sample_dict["line_number"] = i
        sample_dict["total_lines"] = total_lines_in_sample -1
        sample_lines.append(sample_dict)

    df = pd.DataFrame(sample_lines)
    df["line_number_total"] = df["line_number"].astype(str) + "_of_" + df["total_lines"].astype(str)
    line_number_total_encoded = one_hot.transform(np.expand_dims(df["line_number_total"], axis=1))
    line_number_total_encoded = tf.cast(line_number_total_encoded.toarray(), dtype=tf.int32)
    return (abstract_lines, abstract_chars, line_number_total_encoded)

@app.post('/predict')
async def predict_salary(data:RctAbstract):
    abstract = data.abstract

    a, b, c = transform_abstract(abstract)
    prediction = model.predict(x=(tf.constant(a), tf.constant(b), c))
    
    # Turn prediction probs to pred class
    abstract_preds = tf.argmax(prediction, axis=1)

    return abstract_preds
# { 'prediction': prediction[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8001)
