import tensorflow as tf

thresholds = [0.1, 0.15, 0.2]

for threshold in thresholds:
    print(f"threshold: {threshold}")
    model_name = "mut_models/mut_NN_t" + str(threshold)
    model = tf.keras.models.load_model(model_name)
    print(f"mut_model: {str(model.input_shape[1])}")
    model.summary()
    print("\n\n")
    
    model_name = "pop_models/pop_NN_t" + str(threshold)
    model = tf.keras.models.load_model(model_name)
    print(f"pop_model: {str(model.input_shape[1])}")
    model.summary()
    print("\n\n")
    
    model_name = "rec_models/rec_NN_t" + str(threshold)
    model = tf.keras.models.load_model(model_name)
    print(f"rec_model: {str(model.input_shape[1])}")
    model.summary()
    print("\n\n")
