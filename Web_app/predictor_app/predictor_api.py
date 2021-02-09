import pickle as pkl 
import numpy as np 

with open('./models/rfc3.pkl', 'rb') as f:
	rfc3 = pkl.load(f)

feature_names = rfc3.feature_names

def make_prediction(feature_dict):
	"""
	Input:
	feature_dict: a dictionary of the form {"feature_name": "value"}

	Function makes sure the features are fed to the model in the same order the
	model expects them.

	Output:
	Returns (x_inputs, probs) where
	  x_inputs: a list of feature values in the order they appear in the model
	  probs: a list of dictionaries with keys 'name', 'prob'
	"""
	x_input = []
	for name in rfc3.feature_names:
		x_input_ = float(feature_dict.get(name, 0))
		x_input.append(x_input_)

	pred_probs = rfc3.predict_proba([x_input]).flat

	probs = []
	for index in np.argsort(pred_probs)[::-1]:
		prob = {
			'name': rfc3.target_names[index],
			'prob': round(pred_probs[index], 5)
		}
		probs.append(prob)

	return (x_input, probs)


if __name__ == '__main__':
	from pprint import pprint
	print("Checking to see what setting all params to 0 predicts")
	features = {f: '0' for f in feature_names}
	print('Features are')
	pprint(features)

	x_input, probs = make_prediction(features)
	print(f'Input values: {x_input}')
	print('Output probabilities')
	pprint(probs)