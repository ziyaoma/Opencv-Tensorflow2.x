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


def trainmodel():
    # build data
    input_x = np.random.rand(1000, 4)
    output_y = np.dot(input_x,np.array([[3],[14],[6],[10]]))

    inputs = tf.keras.layers.Input(shape=(4,))
    outputs = tf.keras.layers.Dense(units=1)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),loss='mse')
    model.fit(x=input_x,y=output_y,epochs=10,validation_split=0.1)

    #保存
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

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
                      name="frozen_graph.pb",
                      as_text=False)

def tftest():
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
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

    test_x = np.array([[1, 1, 1, 1]], np.float32)
    pred_y = frozen_func(x=tf.constant(test_x,dtype=tf.float32))[0]
    print(pred_y)
    true_y = np.dot(test_x, np.array([[3], [14], [6], [10]]))
    print(true_y)  # [[33.]]


def opencvtest():
    test_x = np.array([[1, 1, 1, 1]], np.float32)
    net = cv2.dnn.readNetFromTensorflow("./frozen_models/frozen_graph.pb")
    net.setInput(test_x)
    pred = net.forward()
    print(pred)

if __name__ == "__main__":
    trainmodel()
    tftest()
    opencvtest()
