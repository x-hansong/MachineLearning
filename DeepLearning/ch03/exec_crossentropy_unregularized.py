import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network_l2
net = network_l2.Network([784, 30, 10], cost=network_l2.CrossEntropyCost)

# setting regularization punished parameter lmbda = 0.0 means no regularization
net.SGD(training_data, 5, 10, 0.5,
        lmbda=0.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=False,
        monitor_training_cost=False
)

net.save("crossentropy_unregularized_net.json")