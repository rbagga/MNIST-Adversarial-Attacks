import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from os import path

### DO NOT CHANGE THE FOLLOWING CONFIGURATION###
######################################################################
TARGET_MODEL_PATH = "../model/mnist_nn.h5"
EPSILON = 0.01
EPOCH = 20
ITERATION = 40 

np.random.seed(0)
tf.set_random_seed(0)
######################################################################

def stratified_split(X, y, ratio):

    """
    Split dataset (X, y) into (X_select, y_select) and (X_unselect, y_unselect).
    The split should be a stratified one.
    The size of selected data should be the ratio times the size of data to be split
    :param X: Feature vectors of data to be split.
    :param y: Labels of data to be split.
    :param ratio: Ratio of data to be selected.
    :return X_select: Feature vectors of selected data.
    :return y_select: Labels of selected data.
    :return X_unselect: Features of unselected data.
    :return y_unselect: Labels of unselected data.
    """
    
    # TODO: Implement the stratified split
    # HINT: Consider train_test_split you used in homework 1.
    X_select, X_unselect, y_select, y_unselect = train_test_split(X, y, test_size = 1.0-ratio, random_state =0, stratify = y)
    print("X select len: ", len(X_select))
    print("total len: ", len(X))
    return X_select, y_select, X_unselect, y_unselect

def train(X_train, y_train):

    """
    Build mnist classifier.
    :param X_train: Feature vectors of training data. Three-dimensional array.
    :param y_tain: Labels of training data. One-dimensional array.
    :return model: Classification model.
    """
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #stratified split? 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = 'softmax') 
    ])
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', metrics=['accuracy'], loss = 'sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=20, batch_size=128)







    # TODO: Build a classifier on the training data.
    return model

def overall_evaluate(model, X_test, y_test):

    """
    Evaluate a classifier on test data. Print the accuracy on test data.
    :param model: Mnist classifer
    :param X_test: Feature vectors of test data. Three-dimensional array.
    :param y_test: Labels of test data. One-dimensional array.
    """
    #model.evaluate(X_test, y_test, verbose=0)[1] 
    # TODO: Get the overall accuracy of the model on the test data, then print.
    # No value to be returned.
    print("overall acc: ")
    predict = model.predict(X_test)
    wrong = np.where(predict != y_test)
    print(1.0 - float(len(wrong))/float(len(y_test)))
	
def class_evaluate(model, X_test, y_test):

    """
    Evaluate a classifier on test data. Print the accuracy for each class of the test data.
    :param model: Mnist classifer
    :param X_test: Feature vectors of test data. Three-dimensional array.
    :param y_test: Labels of test data. One-dimensional array.
    """
    y_test_pred = model.predict(X_test)
    y_test_max=model.predict(X_test).argmax(axis=-1)
    #y_test_max = np.argmax(y_test_pred, axis=1)
	
    print("report:")
    print(classification_report(y_test, y_test_max))
    # TODO: Get the classification performance on each class of the test data, then print.
    # The predicted label of X_test should be an integer instead of its one-hot encoding. 
    # No value to be returned.


def PGD(target_model, X_seed, y_seed, epsilon, epoch, sess):
    print("X_seed len: ", len(X_seed))
    """
    Implementation of the PGD (projected gradient descent) attack.
    :param target_model: The target model to be evaded.
    :param X_seed: Feature vectors of evasion seeds. Three-dimensional array.
    :param y_seed: Labels of evasion seeds. One-dimensional array.
    :param epsilon: Step-size of the PGD attack.
    :param epoch: Number of PGD epochs.
    :param sess: Session for running TensorFlow operations.
    :return X_adv: Adversarial examples of PGD. Three-dimensional array.    
    """

    # TODO: Implement PGD and return X_adv, the adversarial examples produced.
    # TODO: Evaluate the overall accuracy of the of the model on the adversarial example in each epoch.
    output = target_model.output
    #output = target_model.predict(X_seed)
    y_true = tf.placeholder("uint8", [None])
    loss = tf.keras.backend.sparse_categorical_crossentropy(y_true, output)
    grad = tf.gradients(loss, target_model.input)
    Xi  = tf.clip_by_value((target_model.input  + epsilon * tf.math.sign(grad)), 0, 1)
    #Xi  = tf.clip_by_value(Xi, 0, 1)
    #X_adv = Non
    image_count = len(X_seed)
    #X_adv = np.zeros((image_count, 28, 28))
    accl = []
    for i in range(epoch):
		X_adv = np.zeros((image_count, 28, 28))

		if (i == 0):
			X_i = X_seed
		else:
			X_i = sess.run(Xi, feed_dict = {y_true : y_seed, target_model.input:X_i})[0]
			X_adv += X_i
		y_adv_pred = target_model.predict(X_adv).argmax(axis=-1)
		adv_invasive_ind = np.where(y_adv_pred != y_seed)[0]
		attack_pct = float(len(adv_invasive_ind))/float(image_count)
		print("attack_pct: ", attack_pct);
		acc = 1.0 - attack_pct 
		accl.append(acc)
		print("PGD ATTACK SCORE: ", len(adv_invasive_ind))
    #plt.plot(np.arange(1, epoch+1), accl)
    #plt.xticks(np.arange(1, epoch+1))
    #plt.yticks((np.arange(10))/10.0)
    #plt.xlabel("pgd epochs")
    #plt.ylabel("model accuracy")
    #plt.title("MODEL ACCURACY OVER PGD EPOCHS")
    #plt.savefig("pgd_attack.png")
    return X_adv	

def iterative_retrain(target_model_path, X_train, y_train, X_def_seed, y_def_seed, 
                        iteration, epsilon, epoch, sess):
    """
    Implementation of iterative retraining by using PGD to produce adversarial exmaples
    :param target_model_path: The path to the target model
    :param X_train: Feature vectors of original training data. Three-dimensional array.
    :param y_train: Labels of original training data. One-dimensional array.
    :param X_def_seed: Feature vectors of retraining seeds. Three-dimensional array.
    :param y_def_seed: Labels of retraining seeds. One-dimensional array.
    :param iteration: Number of retraining iterations.
    :param epsilon: Step-size of PGD attack.
    :param epoch: Number of PGD epochs.
    :param sess: Session for running TensorFlow operations.
    :return retrain_model: The retrained model. 
    """
    retrain_model = tf.keras.models.load_model(target_model_path)
    accu_iteration = []

    	# TODO: Iteratively retrain the target classifier by using PGD to produce adversarial examples.
    	# Then, add those examples into the training data and retrain the classifier.
    	# TODO: Evaluate the overall accuracy of the of the model on the adversarial example in retraining iteration.
    	# Append the accuracy to accu_iteration
    	                 
	
    # TODO: Plot accu_iteration with respect to number of iterations
    for j in range(iteration):
		print("ITERATION: ", j)
		X_adv = PGD(retrain_model, X_def_seed, y_def_seed, EPSILON, EPOCH, sess)
		y_adv_pred = retrain_model.predict(X_adv).argmax(axis=-1)
		print("size of added", len(X_def_seed))
		adv_invasive_ind = np.where(y_adv_pred != y_def_seed)[0]
		print("wrong: ", len(adv_invasive_ind))
		print("total: ", len(X_def_seed))
		print("wrong ind: ")
		print(adv_invasive_ind)
		wrong = float(len(adv_invasive_ind))/float(len(X_def_seed))
		accuracy = 1.0  - wrong
		print("model new accuracy", accuracy)
		accu_iteration.append(accuracy)
		'''
		for i in adv_invasive_ind:
			adv = X_adv[i]
			X_train = np.vstack((X_train, [adv]))
			#X_train = np.append(X_train, adv)
			#print(X_train.shape)
			#print(adv.shape)
			#X_train = np.concatenate(X_train, adv )
			#X_train = np.append((X_train, adv), axis=0)
			y_train = np.append(y_train, y_adv_seed[i])
		'''
		X_train = np.vstack((X_train, X_adv))
		y_train = np.append(y_train, y_def_seed) 
		print(len(X_train))
		print(len(y_train))
		#retrain_model = train(X_train, y_train)
		retrain_model.fit(X_train, y_train, epochs=20)
    plt.plot(np.arange(iteration), accu_iteration)
    plt.xlabel("retraining iterations")
    plt.ylabel("model accuracy")
    plt.title("model accuracy over retraining iterations")
    plt.savefig("iter_retrain.png")
    #print(accu_iteration)
    print("OVERALL EVALUATE RETRAIN")
    print("_______")
    print("_______")
    overall_evaluate(retrain_model, X_def_seed, y_def_seed)
    print("CLASS EVALUATE RETRAIN")
    class_evaluate(retrain_model, X_def_seed, y_def_seed)
    return retrain_model    

def embed_backdoor(X, y, ratio):

	"""
    Add backdoors to a ratio of "stratified" subset of data (X, y).
    Return the posioned and un-poisoned subsets of data.
    :param X: Feature vectors of data to be poisoned.
    :param y: Labels of data to be poisoned.
    :param ratio: Ratio of data to be contaminated.
    :return X_poison: Feature vectors of contaminated subset of data.
    :return y_poison: Labels of contaminated subset of data.
    :return X_clean: Feature vectors of un-contaminated subset of data.
	:return y_clean: Labels of un-contaminated subset of data 
	"""

	# Split the whole data into clean and poison data
	X_poison, y_poison, X_clean, y_clean = stratified_split(X, y, ratio)
	

    # TODO: Add backdoors to X_poison and change the corresponding label y_poison
	for i in range(len(X_poison)):
		X_poison[i][26][26] = 1
		y_poison[i] += 1
		y_poison[i] = y_poison[i] % 10
	return X_poison, y_poison, X_clean, y_clean

def trojan_train(X_train, y_train, ratio):

    """
    Implementation of the training phase of Trojan attack.
    :param X_train: Feature vectors of the original training data.
    :param y_train: Lables of the original training data.
    :param ratio: Ratio of training data to be contaminated.
    :return poison_model: Classification model obtained by using trojan attack.
    """

    X_train_poison, y_train_poison, X_train_clean, y_train_clean = embed_backdoor(X_train, y_train, ratio)
    
    # Build the poisoned classifier on the poisoned training data

    # TODO: Get the new training data, the union of (X_train_poison, y_train_poison) and (X_train_clean, y_train_clean)
    X_train = np.vstack((X_train_poison, X_train_clean))
    print(y_train_poison.shape)
    print(y_train_clean.shape)
    #y_train = np.vstack((y_train_poison, y_train_clean))
    y_train = np.append(y_train_poison, y_train_clean)
    print(y_train)	
    # TODO: Shuffle the training data (X_train, y_train)
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Get the poisoned model
    poison_model = train(X_train, y_train)

    # Save some contaminated figures
    if ratio == 0.05:
        X_to_save = X_train_poison[0:10]
        y_to_save = y_train_poison[0:10]
        for i in range(10):
            img = X_to_save[i].reshape(28,28)
            plt.imshow(img, cmap = 'gray')
            plt.savefig("../result/trojan_example/Figure{}-{}.png".format(i, y_to_save[i]))        
    return poison_model

def trojan_evaluate(poison_model, X_test, y_test, ratio):

    """
    Implementation of the evaluation phase of Trojan attack.
    :param poison_model: Classification model obtained b/y using trojan attack.
    :param X_test: Feature vectors of the original test data.
    :param y_test: Labels of the original test data.
    :param ratio: Ratio of test data to be contaminated.
    """

    X_test_poison, y_test_poison, X_test_clean, y_test_clean = embed_backdoor(X_test, y_test, ratio)

    # TODO: Evaluate on the clean test data
    # Evaluate with both overall and each class's accuracy
    #print("Evaluate the poisoned model on clean test data:")
    #overall_evaluate(poison_model, X_test_clean, y_test_clean)
    #class_evaluate(poison_model, X_contam, y_contam)
    # TODO: Evaluate on the contaminated test data:
    # Evaluate with both overall and each class's accuracy
    X_contam = np.vstack((X_test_poison, X_test_clean))
    #y_contam = np.vstack((y_test_poison, y_test_clean))
    y_contam = np.append(y_test_poison, y_test_clean)
	#contam_pred = poison_model.predict(X_contam).argmax(axis=-1)[0]
	#wrongidx = np.where(contam_pred != y_contam)
	#comntam_acc = 1.0 - float (len(wrongidx))/ float(len(y_contam))
    print("poisoned model on poisoned test data")
    idx = np.random.permutation(len(X_contam))
    X_contam = X_contam[idx]
    y_contam = y_contam[idx]
    overall_evaluate(poison_model, X_contam, y_contam)
    class_evaluate(poison_model, X_contam, y_contam)
    print("poisoned model on clean test data")
    overall_evaluate(poison_model, X_test_clean, y_test_clean)
    class_evaluate(poison_model, X_test_clean, y_test_clean)
    #print("Evaluate the poisoned model on poisoned test data (effectiveness of attack):")


if __name__ == "__main__":  

    if len(sys.argv) < 2:
        print("python main.py [mode]")
        print("mode: {train, pgd, retrain, trojan}")
        sys.exit(1)
    
    mode = sys.argv[1]
    

    # Get the dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, y_train, _, _ = stratified_split(X_train, y_train, 0.2)
    X_test, y_test, _, _ = stratified_split(X_test, y_test, 0.2)

    # Get the attack seeds
    X_adv_seed = np.copy(X_test)
    y_adv_seed = np.copy(y_test)

    # Get the seeds for iterative retraining
    X_def_seed, y_def_seed, _, _ = stratified_split(X_train, y_train, 0.2)    

    with tf.Session() as sess:
        if mode == 'train':
            # Training
            model = train(X_train, y_train)
            
            # Evaluation
            overall_evaluate(model, X_test, y_test)
            class_evaluate(model, X_test, y_test)
            
            # Save the target model
            model.save(TARGET_MODEL_PATH)

        elif mode == 'pgd':
            if not path.exists(TARGET_MODEL_PATH):
                print("Error! Train your target classifier first.")
                sys.exit(1)
            else:
                # Load the target model
                target_model = tf.keras.models.load_model(TARGET_MODEL_PATH)

                # Generate attack
                X_adv = PGD(target_model, X_adv_seed, y_adv_seed, EPSILON, EPOCH, sess)

                # Get and save the evasive images 
                y_adv_pred = target_model.predict(X_adv).argmax(axis=-1)
                adv_invasive_ind = np.where(y_adv_pred != y_adv_seed)[0][0:10]
                for i in adv_invasive_ind:
                    img = X_adv[i].reshape(28,28)
                    plt.imshow(img, cmap = 'gray')
                    plt.savefig("../result/pgd_example/Figure{}-{}-{}.png".format(i, y_adv_seed[i], y_adv_pred[i]))

        elif mode == 'retrain':
            if not path.exists(TARGET_MODEL_PATH):
                print("Error! Train your target classifier first.")
                sys.exit(1)
            else:
                # Get the retrained model
                retrain_model = iterative_retrain(TARGET_MODEL_PATH, X_train, y_train, X_def_seed, y_def_seed, ITERATION, EPSILON, EPOCH, sess)
                overall_evaluate(retrain_model, X_test, y_test)
                class_evaluate(retrain_model, X_test, y_test)

        elif mode == 'trojan':
            ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            for ratio in ratios:
                print("Ratio of contaminated data: {}".format(ratio))

                # Training phase of Trojan attack
                poison_model = trojan_train(X_train, y_train, ratio)
                # Evaluation phase of Trojan attack
                trojan_evaluate(poison_model, X_test, y_test, ratio)
				
