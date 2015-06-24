import numpy

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


if __name__ == "__main__":
	feat_space = generate_feature_space("train.positive", "train.negative", "test.positive", "test.negative")
	positive_vectors = generate_feat_vectors("train.positive", 1, feat_space)
	print len(positive_vectors)

