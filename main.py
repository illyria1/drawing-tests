from plot import *
from load import *
from core import *

if __name__ == '__main__':
    test_data_frame = load_data('C:/thesis/data/20191012-141556-21001-multicube-MS.json')
    new_data_frame = add_velocity(test_data_frame)
    acceleration_vectors = get_acceleration_vectors(new_data_frame)
    #print(new_data_frame.shape[0])
    plot_2d(test_data_frame, color="pressure")
    for i in range(0, new_data_frame.shape[0], 100):
        plot_vector(new_data_frame.iloc[i], acceleration_vectors[i])
    #plot_vector(new_data_frame.iloc[50], acceleration_vectors[50])
    plt.xlim(100,800)
    plt.show()
    #plot_3d(test_data_frame)
    #plot_3d(test_data_frame, color="pressure")
    input()