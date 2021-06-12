from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    model = load_model("dqn_model.h5")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model.predict(np.array([10]))[0], color="blue")
    ax.plot(model.predict(np.array([0]))[0], color="green")
    ax.plot(model.predict(np.array([-10]))[0], color="red")
    plt.show()
    
    np.argmax(model.predict(np.array([10]))[0])
    
    
if __name__ == '__main__':
    main()