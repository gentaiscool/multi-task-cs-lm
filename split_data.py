import util.datahelper as datahelper
import util.texthelper
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

interview_phase1_data, conversation_phase1_data, all_data, vocab = datahelper.read_seame_phase1()
print("interview_phase1_data speaker:", len(interview_phase1_data.keys()))
print("conversation_phase1_data speaker:", len(conversation_phase1_data.keys()))

valid_conversation_phase1 = ["NC01", "NC02", "NC29", "NC30", "NC42", "NC61"]
valid_interview_phase1 = ["UI01", "UI09"]

test_conversation_phase1 = ["NC09", "NC10", "NC33", "NC34", "NC39", "NC40"]
test_interview_phase1 = ["NI01", "NI03"]

train_data = []
valid_data = []
test_data = []

num_train_speaker = 0
num_valid_speaker = 0
num_test_speaker = 0

for key in interview_phase1_data:
	is_valid = False
	is_test = False

	for i in range(len(valid_interview_phase1)):
		if valid_interview_phase1[i] == key:
			is_valid = True
			print("interview phase1 valid_key:", key)
			break

	for i in range(len(test_interview_phase1)):
		if test_interview_phase1[i] == key:
			is_test = True
			print("interview phase1 test_key:", key)
			break

	for i in range(len(interview_phase1_data[key])):
		if is_valid:
			valid_data.append(interview_phase1_data[key][i])
		else:
			if is_test:
				test_data.append(interview_phase1_data[key][i])
			else:
				train_data.append(interview_phase1_data[key][i])

	if is_valid:
		num_valid_speaker += 1
	if is_test:
		num_test_speaker += 1
	if not is_valid and not is_test:
		num_train_speaker += 1

for key in conversation_phase1_data:
	is_valid = False
	is_test = False

	for i in range(len(valid_conversation_phase1)):
		if valid_conversation_phase1[i] == key:
			is_valid = True
			print("conversation phase1 valid_key:", key)
			break

	for i in range(len(test_conversation_phase1)):
		if test_conversation_phase1[i] == key:
			is_test = True
			print("conversation phase1 test_key:", key)
			break

	for i in range(len(conversation_phase1_data[key])):
		if is_valid:
			valid_data.append(conversation_phase1_data[key][i])
		else:
			if is_test:
				test_data.append(conversation_phase1_data[key][i])
			else:
				train_data.append(conversation_phase1_data[key][i])

	if is_valid:
		num_valid_speaker += 1
	if is_test:
		num_test_speaker += 1
	if not is_valid and not is_test:
		num_train_speaker += 1

print("################################")
print("train_data utterances:", len(train_data))
print("valid_data utterances:", len(valid_data))
print("test_data utterances:", len(test_data))

with open(dir_path + "/data/seame_phase1/train.txt", "w+") as file:
	for i in range(len(train_data)):
		file.write(train_data[i] + "\n")

with open(dir_path + "/data/seame_phase1/valid.txt", "w+") as file:
	for i in range(len(valid_data)):
		file.write(valid_data[i] + "\n")

with open(dir_path + "/data/seame_phase1/test.txt", "w+") as file:
	for i in range(len(test_data)):
		file.write(test_data[i] + "\n")
