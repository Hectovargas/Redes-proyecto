import numpy as np
import matplotlib.pyplot as plt

#def show_error(images,labels):
#    plt.imshow(#imagen, cmap=grey)
#    plt.show()

def show_error(images,labels,y_pred, num_errors = 10):
    pred = np.argmax(y_pred, axis=1)
    errors = np.where(pred != labels)[0]
    plt.figure(figsize=(15, 5))
    
    for i in range(min(num_errors, len(errors))):
        idx = errors[i]
        plt.subplot(2, 5, i + 1)
        
        plt.imshow(images[idx], cmap='gray')
        
        idx = errors[i]      
        plt.title(f"Real: {labels[idx]}\nPred: {pred[idx]}") 
    
    plt.tight_layout()
    plt.show()



def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)

    return np.mean(predictions == y_true)
