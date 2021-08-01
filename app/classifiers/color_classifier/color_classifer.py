import datetime
import base64
import json
import os
from io import BytesIO
import pandas as pd
import numpy as np
import PIL.Image as img
from colorthief import ColorThief
from keras.models import load_model
from pathlib import Path

class ColorClassifier:
    def __init__(self):
        self.path_to_artifacts = "app/classifiers/color_classifier/data/"
        self.path_to_save = "app/classifiers/color_classifier/data/usage/"
        self.index=["color","color_name","hex","R","G","B"]
        self.col_lib_csv = pd.read_csv(self.path_to_artifacts+'colors_library.csv', names=self.index, header=None)
        #load saved classifier model, including its weights and the optimizer
        self.model = load_model(self.path_to_artifacts+'classifier_model.h5')

    def base64_to_image(self, base64_str):
        date = datetime.datetime.now()
        folder = date.strftime("%d_%m_%Y_%H-%M")
        image_data = BytesIO(base64_str)
        img_data = img.open(image_data)
        new_image_data=img_data.resize((100,100))
        image_path=Path(self.path_to_save, folder)
        image_path.mkdir(parents=True, exist_ok=True)
        image_data_path=self.path_to_save + folder + '/kobimdi.png'
        if image_path:
            new_image_data.save(image_data_path, "PNG")
            new_image_data_path= Path(image_data_path)
        return str(new_image_data_path)
      

    def color_stripping(self, input_data):
        print(input_data)
        colorthief=ColorThief(input_data)
        dmtcolor=colorthief.get_color(quality=1)
        arr = dmtcolor
        r,g,b = arr
        return r,g,b

    def preprocessing(self, R,G,B):
            minimum = 10000
            for i in range(len(self.col_lib_csv)):
                d = abs(R- int(self.col_lib_csv.loc[i,"R"])) + abs(G- int(self.col_lib_csv.loc[i,"G"]))+ abs(B- int(self.col_lib_csv.loc[i,"B"]))
                if(d<=minimum):
                    minimum = d
                    cname = self.col_lib_csv.loc[i,"color_name"]
            return cname
        
    def predict(self, input_data):
        pred = self.model.predict(input_data)
        predicted_encoded_train_labels = np.argmax(pred, axis=1)
        return predicted_encoded_train_labels

    def postprocessing(self, input_data):
        colour_value = []
        for row in input_data:
            if row == 0:
                colour_value.append("Red")
            elif row == 1:
                colour_value.append("Green")
            elif row == 2:
                colour_value.append("Blue")
            elif row == 3:
                colour_value.append("Yellow")
            elif row == 4:
                colour_value.append("Orange")
            elif row == 5:
                colour_value.append("Pink")
            elif row == 6:
                colour_value.append("Purple") 
            elif row == 7:
                colour_value.append("Brown")
            elif row == 8:
                colour_value.append("Grey") 
            elif row == 9:
                colour_value.append("Black")
            elif row == 10:
                colour_value.append("White")
        return colour_value


    def rgb2hex(self,r,g,b):
        res_hex= '#%02x%02x%02x' % (r,g,b)
        return res_hex


    def compute_prediction(self, input_data):
        try:
            img_res_data = self.base64_to_image(input_data)
            r,g,b = self.color_stripping(img_res_data)
            hexcode = self.rgb2hex(r,g,b)
            data = {'r': [r],'g': [g],'b': [b]}
            pred_df = pd.DataFrame(data)
            Colorname = self.preprocessing(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
            prediction = self.predict(pred_df)
            prediction = self.postprocessing(prediction)
            
            spc_color = Colorname.split()[0]
            prediction = {"Color_common": prediction, "Specific colour": spc_color,"rgb_color_code":Colorname.split()[1:5],"hex code":hexcode, "status": "OK"}
            #print(prediction)
        except Exception as e:
            return {"status": "Prediction Error", "message": str(e)}

        return prediction
