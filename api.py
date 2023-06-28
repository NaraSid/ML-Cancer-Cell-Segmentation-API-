# added fastapi to tensor env
# added uvicorn to tensor env
# added python-multipart to tensor env
# added gunicorn to tensor env
# added opencv-python to tensor env

from fastapi import FastAPI, File, UploadFile
import tensorflow as tensor
import json
from model_definition import SegmentationModel

app = FastAPI()
# model
model = SegmentationModel().model
model.load_weights('cancer_weights.h5')


# route
@app.post('/')
async def scoring_endpoint(data: UploadFile = File(...)):  # coroutine func:multipoint entry
    img_bytes = await data.read()
    img = tensor.io.decode_image(img_bytes)
    y_hat = model.predict(tensor.expand_dims(img, axis=0))
    return {"prediction": json.dumps(y_hat.tolist())}

#test with uvicorn, add "/docs#/" to ip get  GUI, get cURL
# to test in python script, uise Postman API to get python request by importing cURl
#proceed in test.ipynb