import pandas as pd
import argparse
import joblib
import json
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from keras.models import load_model
import json

claves_ordenadas=["audio_62","audio_585","audio_563","audio_682","audio_312","audio_773","audio_45","audio_279","audio_302","audio_737","audio_276","audio_256","audio_505","audio_111","audio_937","audio_599","audio_129","audio_739","audio_940","audio_852","audio_652","audio_845","audio_421","audio_7","audio_186","audio_902","audio_629","audio_992","audio_624","audio_829","audio_372","audio_897","audio_972","audio_199","audio_326","audio_559","audio_840","audio_277","audio_738","audio_683","audio_114","audio_962","audio_752","audio_449","audio_423","audio_376","audio_269","audio_263","audio_834","audio_105","audio_895","audio_553","audio_349","audio_486","audio_881","audio_908","audio_710","audio_839","audio_288","audio_946","audio_76","audio_990","audio_22","audio_452","audio_37","audio_512","audio_917","audio_290","audio_745","audio_974","audio_160","audio_77","audio_102","audio_673","audio_666","audio_293","audio_32","audio_983","audio_379","audio_301","audio_179","audio_945","audio_755","audio_72","audio_555","audio_454","audio_762","audio_462","audio_437","audio_492","audio_500","audio_976","audio_97","audio_497","audio_978","audio_358","audio_647","audio_474","audio_557","audio_243","audio_222","audio_9","audio_507","audio_304","audio_57","audio_242","audio_214","audio_360","audio_800","audio_566","audio_264","audio_298","audio_126","audio_911","audio_933","audio_365","audio_532","audio_680","audio_311","audio_60","audio_351","audio_173","audio_24","audio_128","audio_759","audio_736","audio_416","audio_857","audio_67","audio_860","audio_679","audio_900","audio_213","audio_398","audio_10","audio_106","audio_903","audio_969","audio_799","audio_12","audio_383","audio_43","audio_827","audio_994","audio_970","audio_579","audio_335","audio_546","audio_481","audio_334","audio_472","audio_622","audio_260","audio_792","audio_801","audio_327","audio_586","audio_805","audio_151","audio_998","audio_965","audio_789","audio_109","audio_41","audio_320","audio_443","audio_971","audio_491","audio_70","audio_867","audio_100","audio_156","audio_369","audio_235","audio_91","audio_848","audio_422","audio_959","audio_66","audio_473","audio_580","audio_874","audio_192","audio_350","audio_153","audio_251","audio_950","audio_48","audio_441","audio_938","audio_963","audio_573","audio_445","audio_440","audio_461","audio_756","audio_300","audio_198","audio_720","audio_424","audio_88","audio_678","audio_468","audio_64","audio_484","audio_391","audio_583","audio_233","audio_231","audio_316","audio_677","audio_875","audio_669","audio_914","audio_130","audio_460","audio_36","audio_133","audio_613","audio_92","audio_610","audio_692","audio_413","audio_287","audio_185","audio_96","audio_705","audio_732","audio_615","audio_626","audio_967","audio_436","audio_778","audio_802","audio_569","audio_636","audio_325","audio_142","audio_504","audio_842","audio_560","audio_671","audio_606","audio_44","audio_617","audio_958","audio_471","audio_426","audio_29","audio_587","audio_400","audio_502","audio_650","audio_715","audio_853","audio_209","audio_284","audio_139","audio_764","audio_859","audio_418","audio_921","audio_200","audio_386","audio_359","audio_315","audio_888","audio_658","audio_125","audio_936","audio_346","audio_161","audio_146","audio_357","audio_406","audio_230","audio_691","audio_993","audio_531","audio_923","audio_210","audio_731","audio_744","audio_124","audio_638","audio_241","audio_457","audio_394","audio_696","audio_478","audio_31","audio_363","audio_625","audio_689","audio_984","audio_684","audio_295","audio_700","audio_887","audio_345"]

def load_data(file_path):
    df= pd.read_pickle(file_path)
    # TODO: Load test data from CSV file
    return df

def load_modelo(model_path):
    model = load_model(model_path)
    return model

def make_predictions(df, model):
    
    probabilidades = model.predict(np.array(df["x"].tolist()))
    predictions = np.argmax(probabilidades, axis=1)
    idx= df["idx"].tolist()
    return predictions,idx

def save_predictions(predictions, predictions_file,idx):
    # TODO: Save predictions to a JSON file
    audio_names = idx
    audio_predictions = {audio_names[i]: int(prediction) for i, prediction in enumerate(predictions)}
    dic_final = {"target": audio_predictions}
    diccionario_completo = {}

    for clave in claves_ordenadas:
        valor = dic_final["target"][clave]  # Obtener el valor del diccionario desordenado para la clave actual
        diccionario_completo[clave] = valor

    diccionario_completo = {"target": diccionario_completo}

    with open(predictions_file, 'w') as f:
        json.dump(diccionario_completo, f)

    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Automated Instrument Sound Recognition Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data/test_data.pkl',
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='models/model.h5',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json',
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_modelo(model_file)
    predictions,idx = make_predictions(df, model)
    save_predictions(predictions, output_file,idx)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)
