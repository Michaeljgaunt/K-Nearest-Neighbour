import numpy
from scipy.spatial import distance
import random

def get_features(fname):
	features = set()
	with open(fname) as feat_file:
		for line in feat_file:
			for feat in line.strip().split():
				features.add(feat)
	return features

def generate_feature_space(positive_training_file, negative_training_file, positive_test_file, negative_test_file):
	#Defining feature_space and unioning with all other training/test files.
	feature_space = get_features(positive_training_file)
	feature_space = feature_space.union(get_features(negative_training_file))
	feature_space = feature_space.union(get_features(positive_test_file))
	feature_space = feature_space.union(

		get_features(negative_test_file))
	feature_space = list(feature_space)
	#Instantiating a dictionary to hold the feature space.
	feature_space_dict = {}
	#Iterating through the feature space list and putting it into a dictionary.
	for (feature_id, feature_value) in enumerate(feature_space):
	    feature_space_dict[feature_value] = feature_id
	#Returning the feature space dictionary and the size of the feature space.
	return feature_space_dict

def generate_feat_vectors(filename, label, feat_space):
    review_vectors = []
    #Opening the file passed in the function arguments.
    with open(filename) as file:
        for line in file:
            #A temporary list is made to hold the current vector.
            temp_vector = numpy.zeros(len(feat_space), dtype=float)
            for word in line.strip().split():
                #The feature space index of the currently iterating word is saved
                feature_index = feat_space[word]
                #The element in the temporary vector at that index is changed to a 1 to indicate the word's presence in the review.
                temp_vector[feature_index] = 1
            #The temporary vector is appended as an entry in the review_vector list.
            review_vectors.append((temp_vector, label))
    return review_vectors

def calculate_similarity(vector_x, vector_y):
	return distance.euclidean(vector_x, vector_y)

def predict_label(training_data, test_review, k):
	similarity_scores = []
	for (training_review, label) in training_data:
	    similarity_scores.append((label, calculate_similarity(training_review, test_review)))
	similarity_scores.sort(lambda x, y: 1 if x[1] > y[1] else -1)
	pos_count = neg_count = 0
	for top in similarity_scores[:k]:
	    if top[0] == 1:
	        pos_count += 1
	    else:
	        neg_count += 1
	if pos_count > neg_count:
	    return 1
	else:
	    return -1
	pass

def check_prediction(test_review, predicted_label):
	if(test_review[-1] == predicted_label):
		return True
	else:
		return False

if __name__ == "__main__":
	print "\nGenerating feature space..."
	feat_space = generate_feature_space("train.positive", "train.negative", "test.positive", "test.negative")
	print "Generating training data..."
	training_review_vectors = generate_feat_vectors("train.positive", 1, feat_space)
	training_review_vectors.extend(generate_feat_vectors("train.negative", -1, feat_space))
	print "Generating test data..."
	test_review_vectors = generate_feat_vectors("test.positive", 1 ,feat_space)
	test_review_vectors.extend(generate_feat_vectors("test.negative", -1, feat_space))
	print "Shuffling test data..."
	random.shuffle(test_review_vectors)
	correct_predictions = 0
	total_predictions= 0
	print "Predicting test data labels..."
	for (review_number,(test_review_vector, actual_label)) in enumerate(test_review_vectors):
		total_predictions += 1
		predicted_label = predict_label(training_review_vectors, test_review_vector, actual_label)
		if(predicted_label == 1):
			print "Test review", review_number, ": positive sentiment predicted"
		else:
			print "Test review", review_number, ": negative sentiment predicted"
		if actual_label == predicted_label:
			correct_predictions += 1
	print "\nPrediction accuracy =", (float(correct_predictions) / float(total_predictions)) * float(100)


	
