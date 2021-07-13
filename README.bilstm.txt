to run bilstmTrain:
	- in the same directory need to have 'process_data_3.py', 'train_biLSTM.py', 'bilstm.py', 'biLSTM_chars.py', 'biLSTM_sufpref.py', biLSTM_concat.py'
	- pos/ner directory for data should be in the working directory
	- arguments to main:
		a. repr = {'a','b','c','d'}
		b. trainFile = path to train data 
		c. modelFile = path to file where model is saved
		d. isPos = 1 if the data os POS, 0 if data is NER

	example:
		python3 bilstmTrain a ./pos/train ./model/modelFile 1

to run bilstmPredict:
	- in the same directory need to have 'process_data_3.py', 'bilstm.py', 'biLSTM_chars.py', 'biLSTM_sufpref.py', biLSTM_concat.py'
	- pos/ner directory for data should be in the working directory
	- arguments to main:
		a. repr = {'a','b','c','d'}
		b. trainFile = path to train data 
		c. modelFile = path to file where model is saved
		d. isPos = 1 if the data os POS, 0 if data is NER

	example:
		python3 bilstmPredict a ./pos/train ./model/modelFile 1
