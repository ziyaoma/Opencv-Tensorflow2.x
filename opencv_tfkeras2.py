import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import cv2

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def get_fashion_mnist_data():

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,test_labels) = fashion_mnist.load_data()
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def trainmodel2():
    tf.random.set_seed(seed=0)

    # Get data
    (train_images, train_labels), (test_images,
                                   test_labels) = get_fashion_mnist_data()

    # Create Keras model
    model = keras.Sequential(layers=[
        keras.layers.InputLayer(input_shape=(28, 28), name="input"),
        keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
        keras.layers.Dense(128, activation="relu", name="dense"),
        keras.layers.Dense(10, activation="softmax", name="output")
    ], name="FCN")

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train model
    model.fit(x={"input": train_images}, y={"output": train_labels}, epochs=10)

    # Test model
    test_loss, test_acc = model.evaluate(x={"input": test_images},
                                         y={"output": test_labels},
                                         verbose=2)
    print("-" * 50)
    print("Test accuracy: ")
    print(test_acc)

    # Get predictions for test images
    predictions = model.predict(test_images)
    # Print the prediction for the first image
    print("-" * 50)
    print("Example TensorFlow prediction reference:")
    print(predictions[0])

    # Save model to SavedModel format
    tf.saved_model.save(model, "./frozen_models/simple_model")
    #tf.model.save((model, "./frozen_models/simple_model"))
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="simple_frozen_graph.pb",
                      as_text=False)


def tftest():
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/simple_frozen_graph.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    test_x = cv2.imread("1.png",0)
    test_x=cv2.resize(test_x,(28,28))

    pred_y = frozen_func(x=tf.constant(test_x,dtype=tf.float32))[0]
    print(pred_y[0].numpy())


def opencvtest():
    test_x = cv2.imread("1.png",0)
    test_x = cv2.dnn.blobFromImage(image=test_x, scalefactor=1.0, size=(28, 28))
    net = cv2.dnn.readNetFromTensorflow("./frozen_models/simple_frozen_graph.pb")
    net.setInput(test_x)
    pred = net.forward()
    print(pred)

if __name__ == "__main__":
    trainmodel2()
    tftest()
    opencvtest()
