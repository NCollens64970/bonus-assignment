### Part 1

Sos for part 1 the split value is 0.2 because you specify you want a 80/20 split, then I use the fit function to train the model and the predict function to make predictions on the model

### Part 2

You want 4 clusters so n_clusters=4, and then I use fit_predict to actually cluster the data

### Part 3

input_shape is asking for what ever shape was made with x_train, this num_features is provided in the parentheses, then for the learning I rate I looked up a good rate for Adam and found that values between 0.01 and 0.0001 were best, so I chose 0.002 so I could be accurate while still training a little faster

### Part 4

when row is updated you need to make sure that the value it becomes is legal, to do this I choose the max between r-1 and 0, this is so if it is already zero it will just stay at the edge and not go out of bounds
