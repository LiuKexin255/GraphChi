{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "demo_dnn.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/LiuKexin255/GraphChi/blob/master/demo_dnn.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vfBmf4dHl0eI",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        },
        "outputId": "346986a9-b4ca-456c-dc71-3ff01d562362"
      },
      "source": [
        "# try:\n",
        "#   # %tensorflow_version only exists in Colab.\n",
        "#   %tensorflow_version 2.x\n",
        "# except Exception:\n",
        "#   pass\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import tensorflow.keras as keras\n",
        "print(tf.__version__)\n",
        "import os\n",
        "\n",
        "resolver = tf.contrib.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])\n",
        "tf.contrib.distribute.initialize_tpu_system(resolver)\n",
        "strategy = tf.contrib.distribute.TPUStrategy(resolver)"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "1.14.0\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "WARNING: Logging before flag parsing goes to stderr.\n",
            "W0826 17:19:12.555402 140041900570496 lazy_loader.py:50] \n",
            "The TensorFlow contrib module will not be included in TensorFlow 2.0.\n",
            "For more information, please see:\n",
            "  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md\n",
            "  * https://github.com/tensorflow/addons\n",
            "  * https://github.com/tensorflow/io (for I/O related ops)\n",
            "If you depend on functionality not listed there, please file an issue.\n",
            "\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_IWdsSLqnj7P",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "fashion_mnist = tf.keras.datasets.fashion_mnist\n",
        "(x_train,y_train) , (x_test , y_test) = fashion_mnist.load_data()\n",
        "x_train , x_test = x_train.astype(np.float32) /255.0 , x_test.astype(np.float32) / 255.0\n",
        "x_train = x_train.reshape([-1,28,28,1])\n",
        "x_test = x_test.reshape([-1,28,28,1])\n",
        "y_train , y_test = y_train.astype(np.float32) ,y_test.astype(np.float32)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "og501mnuoWuG",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def create_dnn_model() :\n",
        "  dnn_model = keras.models.Sequential([\n",
        "    keras.layers.BatchNormalization(input_shape=(28,28,1)) , \n",
        "    keras.layers.Conv2D(32 , 3 ,padding=\"SAME\" , activation=\"relu\"),\n",
        "    keras.layers.MaxPooling2D(3 , 2 , \"SAME\"),\n",
        "    keras.layers.BatchNormalization() ,\n",
        "    keras.layers.Conv2D(64 , 3 ,padding=\"SAME\" , activation=\"relu\"),\n",
        "    keras.layers.MaxPooling2D(3 , 2 , \"SAME\"),\n",
        "    keras.layers.BatchNormalization() , \n",
        "    keras.layers.Conv2D(128 , 3 ,padding=\"SAME\" , activation=\"relu\"),\n",
        "    keras.layers.MaxPooling2D(3 , 2 , \"SAME\"),\n",
        "    # keras.layers.GlobalAveragePooling2D(),\n",
        "    keras.layers.Flatten() , \n",
        "    keras.layers.BatchNormalization() ,\n",
        "    keras.layers.Dropout(0.5),\n",
        "    keras.layers.Dense(1024 , \"relu\") ,\n",
        "    keras.layers.BatchNormalization(),\n",
        "    keras.layers.Dropout(0.5),\n",
        "    keras.layers.Dense(10 , \"softmax\")\n",
        "\n",
        "  ])\n",
        "  return dnn_model"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xexCx__8rsJp",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "dnn_model = create_dnn_model()\n",
        "dnn_model.compile(optimizer=\"adam\" , loss=\"sparse_categorical_crossentropy\" , metrics=['accuracy'])\n",
        "dnn_model.fit(x_train , y_train,256 , 30)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Gvri90PiwLut",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\n",
        "dnn_model.evaluate(x_test , y_test)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3v49eYwW4dzF",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 91
        },
        "outputId": "fd9e5886-fe36-4f90-85fb-bf4953d1f641"
      },
      "source": [
        "\n",
        "with strategy.scope():\n",
        "  model = create_dnn_model()\n",
        "  model.compile(\n",
        "      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),\n",
        "      loss='sparse_categorical_crossentropy',\n",
        "      metrics=['sparse_categorical_accuracy'])\n",
        "\n"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "W0826 17:19:33.451308 140041900570496 deprecation.py:506] From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/init_ops.py:1251: calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Call initializer instance with the dtype argument instead of passing it to the constructor\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ci1K7Gcb7E6K",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "0585dc97-727b-4fad-ed23-cb4876ac326e"
      },
      "source": [
        "model.fit(\n",
        "    x_train, y_train,epochs=30, steps_per_epoch=60,\n",
        ")\n",
        "model.evaluate(x_test , y_test)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/30\n",
            "60/60 [==============================] - 4s 66ms/step - loss: 0.0850 - sparse_categorical_accuracy: 0.9680\n",
            "Epoch 2/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0793 - sparse_categorical_accuracy: 0.9703\n",
            "Epoch 3/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0797 - sparse_categorical_accuracy: 0.9694\n",
            "Epoch 4/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0769 - sparse_categorical_accuracy: 0.9714\n",
            "Epoch 5/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0736 - sparse_categorical_accuracy: 0.9717\n",
            "Epoch 6/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0734 - sparse_categorical_accuracy: 0.9729\n",
            "Epoch 7/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0701 - sparse_categorical_accuracy: 0.9730\n",
            "Epoch 8/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0686 - sparse_categorical_accuracy: 0.9741\n",
            "Epoch 9/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0679 - sparse_categorical_accuracy: 0.9747\n",
            "Epoch 10/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0619 - sparse_categorical_accuracy: 0.9772\n",
            "Epoch 11/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9773\n",
            "Epoch 12/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0600 - sparse_categorical_accuracy: 0.9777\n",
            "Epoch 13/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0593 - sparse_categorical_accuracy: 0.9777\n",
            "Epoch 14/30\n",
            "60/60 [==============================] - 1s 18ms/step - loss: 0.0582 - sparse_categorical_accuracy: 0.9777\n",
            "Epoch 15/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0536 - sparse_categorical_accuracy: 0.9799\n",
            "Epoch 16/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0556 - sparse_categorical_accuracy: 0.9789\n",
            "Epoch 17/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0522 - sparse_categorical_accuracy: 0.9808\n",
            "Epoch 18/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0514 - sparse_categorical_accuracy: 0.9811\n",
            "Epoch 19/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0484 - sparse_categorical_accuracy: 0.9818\n",
            "Epoch 20/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0515 - sparse_categorical_accuracy: 0.9808\n",
            "Epoch 21/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0471 - sparse_categorical_accuracy: 0.9823\n",
            "Epoch 22/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0474 - sparse_categorical_accuracy: 0.9829\n",
            "Epoch 23/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0452 - sparse_categorical_accuracy: 0.9836\n",
            "Epoch 24/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0443 - sparse_categorical_accuracy: 0.9836\n",
            "Epoch 25/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0451 - sparse_categorical_accuracy: 0.9832\n",
            "Epoch 26/30\n",
            "60/60 [==============================] - 1s 17ms/step - loss: 0.0467 - sparse_categorical_accuracy: 0.9823\n",
            "Epoch 27/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0447 - sparse_categorical_accuracy: 0.9836\n",
            "Epoch 28/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0414 - sparse_categorical_accuracy: 0.9843\n",
            "Epoch 29/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0398 - sparse_categorical_accuracy: 0.9855\n",
            "Epoch 30/30\n",
            "60/60 [==============================] - 1s 16ms/step - loss: 0.0408 - sparse_categorical_accuracy: 0.9847\n",
            "313/313 [==============================] - 7s 21ms/step\n",
            "313/313 [==============================] - 7s 21ms/step\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0.3079383361388367, 0.9259]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    }
  ]
}