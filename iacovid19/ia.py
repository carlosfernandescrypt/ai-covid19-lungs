import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import zipfile

arquivo = open('/content/rn_covid_rx.json', 'r')
er = arquivo.read()
arquivo.close()

classificador = model_from_json(er)
classificador.load_weights('/content/rx_covid_rx.h5')

imagem_teste = image.load_img('/content/covid dataset/test/NORMAL/103.jpeg', target_size = (64,64)) 
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0)
previsao = classificador.predict(imagem_teste)
previsao = (previsao > 0.5)

if previsao == True:
  print('Normal')
else:
  print('Covid')