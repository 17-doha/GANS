from matplotlib import pyplot as plt
def visualize_10(X_train):
    fig, axs = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):  
        # plot image pixesles
            axs[i,j].imshow(X_train[i+j], cmap=plt.get_cmap('gray'))
    # Display the image
    plt.show()