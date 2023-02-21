#!/usr/bin/python
import json
import base64
import shutil
import os
from websocket import create_connection
i=0
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
prediction_model = keras.models.load_model('prediction_model')
def solve_captcha(filepath):
    #Path to the data directory
    data_dir = Path("./captchas/")

    # Get list of all the images
    images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
    labels = [img.split(os.path.sep)[-1].split(".jpg")[0].lower() for img in images]
    characters = set(char for label in labels for char in label)
    characters = sorted(list(characters))
    #print("Number of unique characters: ", len(characters))
    #print("Characters present: ", characters)
    del labels

    images=[filepath]

    # Batch size for training and validation
    batch_size = 1

    # Desired image dimensions
    img_width = 400
    img_height = 90

    # Maximum length of any captcha in the dataset
    max_length = 7

    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )


    # Splitting data into training and validation sets
    x_valid, y_valid = images, np.array([""])

    def encode_single_sample(img_path, label):
        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_png(img, channels=1)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, [img_height, img_width])
        # 5. Transpose the image because we want the time
        # dimension to correspond to the width of the image.
        img = tf.transpose(img, perm=[1, 0, 2])
        # 6. Map the characters in label to numbers
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        # 7. Return a dict as our model is expecting two inputs
        return {"image": img, "label": label}

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
	    encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
	    :, :max_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
            return output_text

    #  Let's check results on some validation samples
    for sample in validation_dataset.take(1):
        images = sample["image"]
        labels = sample["label"]

        preds = prediction_model.predict(images)
        pred_texts = decode_batch_predictions(preds)
        #print(pred_texts)
        orig_texts = []
        for label in labels:
	        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
	        orig_texts.append(label)

        for i in range(len(pred_texts)):
	        img = (images[i, :, :, 0] * 255).numpy().astype(np.uint8)
	        img = img.T
	        title = f"{pred_texts[i]}"
	        print(title)
	        return title
while True:
	i+=1
	print("#",i)
	ws = create_connection("wss://server.6obcy.pl:7010/6eio/?EIO=3&transport=websocket")
	captchareceived=False
	captcha_answer=""
	ceid=1
	while True:
		result = ws.recv()
		data=json.loads(result[1:])
		if "ev_name" in data:
			if data["ev_name"]=="cn_acc":
				conn_payload={}
				conn_payload["ev_name"]="_cinfo"
				conn_payload["ev_data"]={}
				conn_payload["ev_data"]["hash"]=data["ev_data"]["hash"]
				conn_payload["ev_data"]["dpa"]=True
				conn_payload["ev_data"]["caper"]=True
				ws.send("4"+json.dumps(conn_payload))
				ws.send('4{"ev_name":"_sas","ev_data":{"channel":"main","myself":{"sex":0,"loc":0},"preferences":{"sex":0,"loc":0}},"ceid":1}')
				ceid=1
			elif data["ev_name"]=="caprecvsas":
				with open("captcha.jpg", "wb") as captchafile:
					captchafile.write(base64.b64decode(data["ev_data"]["tlce"]["data"][22:]))
				captcha_answer=solve_captcha("captcha.jpg")
				conn_payload={}
				conn_payload["ev_name"]="_capsol"
				conn_payload["ev_data"]={}
				conn_payload["ev_data"]["solution"]=captcha_answer
				ws.send("4"+json.dumps(conn_payload))
				captchareceived=True
			elif data["ev_name"]=="capissol":
				if data["ev_data"]["success"]==True:
					print("Captcha OK")
					shutil.move('captcha.jpg', 'captchas/'+captcha_answer+".jpg")
					break
				if data["ev_data"]["success"]==False:
					print("Captcha FAIL")
					shutil.move('captcha.jpg', 'unsolved/'+captcha_answer+".jpg")
					break
	ws.close()
