import numpy as np
import neural_network as NN
import UI

# Basic Test version/NN
max_categories_used = 20     # Change later
num_used_doodles_per_category = 10000    # It isnt supposed to take forever


# Get the categories names
with open(r"C:\Users\Windows\Documents\Uni\Uni_Bonn\Mathematik\lab_machine_learning\final_project\category_names.txt", 'r') as f:
    category_names = [line.strip() for line in f]
num_cat = len(category_names)
num_cat = min(num_cat, max_categories_used)

print("Number of categories is " , num_cat)

# Get the data into one large array. For each data point, the corresponding label is a unit vector with a one at its category's place.
with open(r"C:\Users\Windows\Documents\Uni\Uni_Bonn\Mathematik\lab_machine_learning\final_project\category_filenames.txt", 'r') as f:
    file_names = [line.strip() for line in f]

x = np.empty((0, 28, 28, 1))
y = np.empty((0,num_cat))

for i in range(num_cat):
    file = file_names[i]

    # Append the data to x
    data = np.load(file=r"C:\Users\Windows\Documents\Uni\Uni_Bonn\Mathematik\lab_machine_learning\final_project\doodles/" + file)
    data = data.reshape((data.shape[0], 28, 28, 1))
    x = np.vstack((x, data[:num_used_doodles_per_category]))

    # Append the labels to y with the ones set correctly
    label = np.zeros((data.shape[0], num_cat))
    label[:, i] = 1
    y = np.vstack((y, label[:num_used_doodles_per_category]))

print("Loading data completed. Number of data points = ", x.shape[0])

# Shuffle x and y randomly (but equally) such that all categories are used as training and test data
permut = np.random.permutation(x.shape[0])
x = x[permut]/255.
y = y[permut]

print(np.sum(y, axis=0))

# Make 10% of the data test data and the other 90% training data
part = x.shape[0]//10
x_test = x[:part]
x_train = x[part:]
y_test = y[:part]
y_train = y[part:]

print('Training data = ', x_train.shape[0])


print("Separating data into test and training sets is complete. Start setting up the neural network.")

test_model = NN.create_model(dim_output=num_cat)


hist = test_model.fit(x_train, y_train, batch_size = 128, epochs = 6, validation_data = (x_test, y_test))
NN.plot_history(hist)
test_model.save_weights(r"c:\Users\Windows\Documents\Uni\Uni_Bonn\Mathematik\lab_machine_learning\final_project\weights\twenty_cat_model.weights.h5")

# test_model.load_weights("test_model.weights.h5")