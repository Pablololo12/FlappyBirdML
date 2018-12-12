from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import getopt
import sys


def netBrain():
    keras = Sequential()
    keras.add(Dense(16, input_shape=(8,), activation='relu'))
    keras.add(Dense(32, activation='relu'))
    keras.add(Dense(8, activation='relu'))
    keras.add(Dense(2, activation='sigmoid'))
    keras.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
    return keras

def main(argv):
    try:
        opts, _ = getopt.getopt(argv,"hr")
    except getopt.GetoptError:
        print("birdML.py [-h | -r]")
        sys.exit(2)
    
    record = False
    for opt, arg in opts:
        if opt == '-h':
            print("-h to help")
            print("-r record")
        elif opt == '-r':
            record = True

    netb = netBrain()
    netb.summary()
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True, force_fps=True)
    p.init()
    actions = p.getActionSet()

    out = 1

    epochs = 50
    for i in range(epochs):
        lstates = []
        rewards = []
        if record:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('Videos/test_'+str(i)+'.mov', fourcc, 30.0, (288, 512))
        for d in range(10):
            while not p.game_over():
                if record:
                    obs = p.getScreenRGB()
                    obs = cv2.transpose(obs)
                    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                    out.write(obs)
                st = game.getGameState()
                gstate = list(st.values())
                gstate = np.array([np.array(gstate)])
                lstates.append(gstate[0])
                pred = netb.predict(gstate)[0]
                a = pred.argmax()
                p.act(actions[a])
                if st['next_pipe_bottom_y'] < st['player_y']:
                    pred[0] = 1.0
                    pred[1] = 0.0
                elif st['next_pipe_top_y'] > st['player_y']:
                    pred[0] = 0.0
                    pred[1] = 1.0
                rewards.append(pred)
            p.reset_game()
        netb.fit(np.array(lstates),
                 np.array(rewards),
                     batch_size=10,
                         epochs=10)
        if record:
            out.release()

if __name__ == '__main__':
    main(sys.argv[1:])