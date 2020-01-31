import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network_l1
net = network_l1.Network([784, 30, 10], cost=network_l1.CrossEntropyCost)

# regularization punished parameter lmbda = 5.0
net.SGD(training_data, 5, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=False,
        monitor_training_accuracy=False,
        monitor_training_cost=False
)

net.save("crossentropy_L1_net.json")