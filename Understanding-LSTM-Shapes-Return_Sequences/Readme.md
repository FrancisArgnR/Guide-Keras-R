# Understanding the LSTM Input_Shapes and Return_Sequences

A brief definition of the dimension of data, the shapes of LSTM layer and the state of return_sequences parameter is done.

## LSTM one to one

Sequences are not processing. A single time step is used in each input.

  ### Dimension of data
  
    The LSTM expected a 3D array as input
  
      x.train -> 3 - > c(batch_size, timestep = 1, features) 
      y.train -> 2 - > c(batch_size, timestep = 1)

  ### Input_Shape

    input_shape = c(timestep,features)

  ### Example

    - LSTM

        model <- keras_model_sequential() <br>
        model %>% <br>
          layer_lstm(units, input_shape, return_sequences = FALSE) %>% <br>
          layer_dense(units) <br>

    - Stacked LSTM

        model <- keras_model_sequential() <br>
        model %>% <br>
          layer_lstm(units, input_shape, return_sequences = TRUE) %>% <br>
          layer_lstm(units, return_sequences = FALSE) %>% <br>
          layer_dense(units) <br>

## LSTM many to one

## LSTM many to many
