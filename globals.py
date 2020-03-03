import json

global MIN_TRAIN_LENGTH
MIN_TRAIN_LENGTH = 5
global MIN_TRANSMISSION_INTERVAL
MIN_TRANSMISSION_INTERVAL = 0.001
global DT_CONSECUTIVE
DT_CONSECUTIVE = 6
global BOUNDARY_PCT
global BOUNDARY_PDT


def update_global_tool():
    with open('globals.json', 'r') as f:
        data = json.load(f)
    global MIN_TRAIN_LENGTH
    global MIN_TRANSMISSION_INTERVAL
    global DT_CONSECUTIVE
    global BOUNDARY_PCT
    global BOUNDARY_PDT
    MIN_TRAIN_LENGTH= data['MIN_TRAIN_LENGTH']
    MIN_TRANSMISSION_INTERVAL = data['MIN_TRANSMISSION_INTERVAL']
    DT_CONSECUTIVE = data['DT_CONSECUTIVE']
    BOUNDARY_PCT = data['BOUNDARY_PCT']
    BOUNDARY_PDT = data['BOUNDARY_PDT']


def update_global_file(mtl, mti, dtc, dpct, dpdt):
    data = {
        "MIN_TRAIN_LENGTH": mtl,
        "MIN_TRANSMISSION_INTERVAL": mti,
        "DT_CONSECUTIVE": dtc,
        "BOUNDARY_PCT": dpct,
        "BOUNDARY_PDT": dpdt
    }
    with open("result.json", "w") as f:
        json.dump(data, f)
