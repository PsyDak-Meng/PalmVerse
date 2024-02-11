import tensorflow as tf

def create_NN_model():
    # Variable-length int sequences.
    query_input = tf.keras.Input(shape=(84,), dtype='int32')
    value_input = tf.keras.Input(shape=(84,), dtype='int32')

    # Embedding lookup.
    token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
    # Query embeddings of shape [batch_size, Tq, dimension].
    query_embeddings = token_embedding(query_input)
    # Value embeddings of shape [batch_size, Tv, dimension].
    value_embeddings = token_embedding(value_input)

    # CNN layer.
    cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=16,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = cnn_layer(query_embeddings)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = cnn_layer(value_embeddings)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    attn_1 = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    dense_output_1 = tf.keras.layers.Dense(units=1024, activation='relu')(attn_1)
    dense_output_1 = tf.keras.layers.Dropout(0.2)(dense_output_1)

    dense_output_1 = tf.keras.layers.Concatenate()([dense_output_1,attn_1])

    #dense_output_1 = self.attn_layer(dense_output_1)
    #dense_output_1 = tf.keras.layers.Concatenate()([input, dense_output_1])
    #dense_output_1 = self.norm_layer(dense_output_1)

    dense_output_2 = tf.keras.layers.Dense(units=512, activation='relu')(dense_output_1)
    #dense_output_2 = tf.keras.layers.Dense(units=256, activation='relu')(dense_output_1)
    dense_output_2 = tf.keras.layers.Dropout(0.2)(dense_output_2)
    #dense_output_2 = self.attn_layer(dense_output_2)
    #dense_output_2 = self.tf.keras.layers.Concatenate()([dense_output_1, dense_output_2])
    # #dense_output_2 = norm_layer(dense_output_2)

    dense_output_3 = tf.keras.layers.Dense(units=256, activation='relu')(dense_output_2)
    dense_output_3 = tf.keras.layers.Dropout(0.2)(dense_output_3)

    dense_output_4 = tf.keras.layers.Dense(units=64, activation='relu')(dense_output_3)
    output = tf.keras.layers.Dense(units=29, activation='softmax')(dense_output_4)

    #self.norm_layer = tf.keras.layers.Normalization()
    model = tf.keras.Model(inputs=(query_input,value_input),outputs=output)
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model

