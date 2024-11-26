import mynetwork3
from mynetwork3 import Network,relu
from mynetwork3 import FullyConnectedLayer , SoftmaxLayer,CrossEntropyLayer,ConvPoolLayer
if __name__ == "__main__":
    training_data , validation_data , test_data = mynetwork3.load_data_shared("data/mnist_expanded.pkl.gz")
    mini_batch_size = 1000
    net = Network ([
            ConvPoolLayer (
            image_shape =( mini_batch_size , 1, 28, 28) ,
            filter_shape =(20 , 1, 5, 5),
            poolsize =(2, 2),
            activation_fn=relu
            ),
            ConvPoolLayer (
            image_shape =( mini_batch_size , 20, 12, 12) ,
            filter_shape =(40 , 20, 5, 5),
            poolsize =(2, 2),
            activation_fn=relu
            ),
            FullyConnectedLayer (n_in =40*4*4 , n_out =100,activation_fn=relu,p_dropout=0.2) ,
            FullyConnectedLayer (n_in =100, n_out =100,activation_fn=relu,p_dropout=0.2) ,
        ],
        SoftmaxLayer (n_in =100 , n_out =10)
    )
    net.SGD(training_data , validation_data , test_data,50,mini_batch_size,0.1,1,0.2 )