import network
import mnist_loader

trainig_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([(28*28), 30, 10])
net.SGD(trainig_data, 3, 10, 3.0, test_data)
print("Average Accuracy: %f" %(net.evaluate(validation_data)/100.0))
i = 0
validationSet = []
while(i<10):
    validationSet.append([(y,z) for (y,z) in validation_data if z == i])
    print("Accuracy of %d's" %i + ": %f" %((net.evaluate(validationSet[i])/float(len(validationSet[i])))*100.0))
    i+=1