# Import the necessary components from Keras
from keras.datasets import fashion_mnist

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


# load labels.csv from datasets folder using pandas
labels = pd.read_csv('dataset/labels.csv',index_col=0)
print('info for labels :',labels.columns)
# print value counts for genus
print(labels['genus'].value_counts())

# assign the genus label values to y
y = labels['genus'].values
print(type(y))

# load an image and explore
example_image = io.imread('dataset/{}.jpg'.format(labels.index[0]))
print(type(example_image))
# show image
# ... YOUR CODE FOR TASK 3 ...
plt.imshow(example_image)
plt.savefig('example_image.png')
# print shape
print('Example image has shape: ', example_image.shape)

# initialize standard scaler
ss = StandardScaler()

image_list = []
for i in labels.index:
    # load image
    img = io.imread('dataset/{}.jpg'.format(i)).astype(np.float64)

    # for each channel, apply standard scaler's fit_transform method
    for channel in range(img.shape[2]):
        img[:, :, channel] = ss.fit_transform(img[:, :, channel])

    # append to list of all images
    image_list.append(img)

# convert image list to single array
X = np.array(image_list)

# print shape of X
print(X.shape)

# split out evaluation sets (x_eval and y_eval)
x_interim, x_eval, y_interim, y_eval = train_test_split(X,
                                           y,
                                           test_size=0.2,
                                           random_state=52)

# split remaining data into train and test sets
# ... YOUR CODE FOR TASK 5 ...
x_train, x_test, y_train, y_test = train_test_split(x_interim,
                                           y_interim,
                                           test_size=0.4,
                                           random_state=52)
# examine number of samples in train, test, and validation sets
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_eval.shape[0], 'eval samples')


n1=2500
n2=n1//5
img_rows=28
img_cols=28

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
print((n1,x_train.shape[1],x_train.shape[2],1))

# The number of image categories
n_categories =max(y_train)+1
categories=np.array(np.unique(y_train))

print(categories,n_categories)
train_data=np.empty((n1,x_train.shape[1],x_train.shape[2],1),dtype=float)
train_labels=np.zeros((n1,n_categories),dtype=float)

for i in range(n1):
    # Find the location of this label in the categories variable
    for jj in range(n_categories):
        if (categories[jj]==y_train[i]):
            #print(jj)
            j=jj

    train_data[i,:,:,0]=x_train[i,:,:]
    train_labels[i,j]=1
print(train_labels[0:3,:])


test_data=np.empty((n2,x_test.shape[1],x_train.shape[2],1),dtype=float)
test_labels=np.zeros((n2,n_categories),dtype=float)

for i in range(n2):
    # Find the location of this label in the categories variable
    for jj in range(n_categories):
        if (categories[jj]==y_test[i]):
            #print(jj)
            j=jj

    test_data[i,:,:,0]=x_test[i,:,:]
    test_labels[i,j]=1
print(test_labels[0,:])

optimization_step=[3]

# fix random seed for reproducibility
seed = 7

np.random.seed(seed)
for case in optimization_step:
    if case == 1:
        # Function to create model, required for KerasClassifier
        def create_model():
            print('')
            print('/////////////////////////////////////  MODEL : DEEPER CONVOULTIONAL NETWORK WITH DROPOUT /////////////')
            print('/////////////////////////////////////  MODEL : EPOCHS,BATCH_SIZE = '+str(epochs)+' '+str(batch_size)+' /////////////')
            # create model
            model=Sequential()
            # Add a convolutional layer
            model.add(Conv2D(15, kernel_size=2, activation='relu',input_shape=(img_rows, img_cols, 1)))
            # Add a dropout layer
            model.add(Dropout(dropout_rate))
            # Add another convolutional layer
            model.add(Conv2D(5,kernel_size=2,activation='relu'))
            # Flatten and feed to output layer
            model.add(Flatten())
            model.add(Dense(n_categories, activation='softmax'))
            model.compile(optimizer=optimizer,metrics=['accuracy'],loss='categorical_crossentropy')
            return model
        # create model
        model = KerasClassifier(build_fn=create_model, verbose=1)
        # define the grid search parameters
        optimizer = 'Nadam'
        batch_size = [10,50,250,500]
        epochs = [2,5,10,15]
        dropout_rate= 0.0
        #Optimize epochs and batch_size
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        # Fit to training data
        grid_result=grid.fit(train_data,train_labels)
    if case == 2:
        # Function to create model, required for KerasClassifier
        def create_model(optimizer='adam'):
            print('')
            print('/////////////////////////////////////  MODEL : DEEPER CONVOULTIONAL NETWORK WITH DROPOUT /////////////')
            print('/////////////////////////////////////  MODEL : Optimizer = '+str(optimizer)+' /////////////')
            # create model
            model=Sequential()
            # Add a convolutional layer
            model.add(Conv2D(15, kernel_size=2, activation='relu',input_shape=(img_rows, img_cols, 1)))
            # Add a dropout layer
            model.add(Dropout(dropout_rate))
            # Add another convolutional layer
            model.add(Conv2D(5,kernel_size=2,activation='relu'))
            # Flatten and feed to output layer
            model.add(Flatten())
            model.add(Dense(n_categories, activation='softmax'))
            model.compile(optimizer=optimizer,metrics=['accuracy'],loss='categorical_crossentropy')
            return model
        # create model
        model = KerasClassifier(build_fn=create_model, verbose=1)
        # define the grid search parameters
        batch_size = 50
        epochs = 10
        dropout_rate= 0.0
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        #Optimize epochs and batch_size
        param_grid = dict(optimizer=optimizer)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        # Fit to training data
        grid_result=grid.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size)
    if case == 3:
        # Function to create model, required for KerasClassifier
        def create_model(dropout_rate=0.0):
            print('')
            print('/////////////////////////////////////  MODEL : DEEPER CONVOULTIONAL NETWORK WITH DROPOUT /////////////')
            print('/////////////////////////////////////  MODEL : DROPOUT_RATE = '+str(dropout_rate)+' /////////////')
            # create model
            model=Sequential()
            # Add a convolutional layer
            model.add(Conv2D(15, kernel_size=2, activation='relu',input_shape=(img_rows, img_cols, 1)))
            # Add a dropout layer
            model.add(Dropout(dropout_rate))
            # Add another convolutional layer
            model.add(Conv2D(5,kernel_size=2,activation='relu'))
            # Flatten and feed to output layer
            model.add(Flatten())
            model.add(Dense(n_categories, activation='softmax'))
            model.compile(optimizer=optimizer,metrics=['accuracy'],loss='categorical_crossentropy')
            return model
        # create model
        model = KerasClassifier(build_fn=create_model, verbose=1)
        # define the grid search parameters
        batch_size = 50
        epochs = 10
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        optimizer = 'Nadam'
        #Optimize epochs and batch_size
        param_grid = dict(dropout_rate=dropout_rate)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        # Fit to training data
        grid_result=grid.fit(train_data,train_labels,epochs=epochs,batch_size=batch_size)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#print(grid_result.params_)
