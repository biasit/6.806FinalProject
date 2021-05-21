# Find the exact match between a set of predictions and target set
def exact_match(pred, val):
  '''
    pred.size() = [number of predictions, length of each prediction]
    val.size() = [number of targets, 1, length of each target]
    number of predictions == number of targets
  '''
  correct = 0
  for i in range(len(pred)):
    wrong = 0
    for j in range(min(len(pred[i]), len(val[i][0]))):
      if pred[i][j] != val[i][0][j]:
        wrong = 1
        break
    if wrong == 0:
      correct += 1
  return correct/len(pred)

# Find the percent well-formed in a prediction set
def percent_well_formed(pred):
  '''
    pred.size() = [number of predictions, length of each prediction]
  '''
  correct = 0
  for i in range(len(pred)):
    paren = 0
    for j in range(len(pred[i])):
      if pred[i][j] == "(":
        paren += 1
      elif pred[i][j] == ")":
        paren -= 1
    if paren == 0:
      correct += 1
  return correct/len(pred)

# Find the list of indices that are not well-formed
def well_formed_list(pred):
  '''
    pred.size() = [number of predictions, length of each prediction]
  '''
  wrong_list = []
  for i in range(len(pred)):
    paren = 0
    for j in range(len(pred[i])):
      if pred[i][j] == "(":
        paren += 1
      elif pred[i][j] == ")":
        paren -= 1
    if paren != 0:
      wrong_list.append(i)
  return wrong_list

# Find the list of indices that are not exact matches
def em_list(pred, val):
  '''
    pred.size() = [number of predictions, length of each prediction]
    val.size() = [number of targets, 1, length of each target]
    number of predictions == number of targets
  '''
  wrong_list = []
  for i in range(len(pred)):
    for j in range(min(len(pred[i]), len(val[i][0]))):
      if pred[i][j] != val[i][0][j]:
        wrong_list.append(i)
        break
  return wrong_list

# Find the common elements between two lists
def common_list(l1, l2):
  '''
    typeof(l1) = list of float, int
    typeof(l2) = list of float, int
  '''
  i=0
  j=0
  common = []
  while i < len(l1) and j < len(l2):
    if l1[i] > l2[j]:
      j += 1
    elif l1[i] == l2[j]:
      common.append(l2[j])
      j += 1
    else:
      i += 1
  return common

# Find the elements in L2 that are not in L1
def uncommon_list(l1, l2):
  '''
    typeof(l1) = list of float, int
    typeof(l2) = list of float, int
  '''
  i=0
  j=0
  uncommon = []
  while i < len(l1) and j < len(l2):
    if l1[i] > l2[j]:
      j += 1
      uncommon.append(l2[j])
    elif l1[i] == l2[j]:
      j += 1
    else:
      i += 1
  return uncommon

# Find the exact match accuracy per length of target output
def accuracy_per_length(pred, val):
  '''
    pred.size() = [number of predictions, length of each prediction]
    val.size() = [number of targets, 1, length of each target]
    number of predictions == number of targets

    returns 
    x = list of lengths
    y = list of accuracies
  '''
  lengths = {}
  correct = {}
  for i in range(len(pred)):
    wrong = 0
    if len(val[i][0]) in lengths:
      lengths[len(val[i][0])] += 1
    else:
      lengths[len(val[i][0])] = 1
      correct[len(val[i][0])] = 0
    for j in range(min(len(pred[i]), len(val[i][0]))):
      if pred[i][j] != val[i][0][j]:
        wrong = 1
        break
    if wrong == 0:
      correct[len(val[i][0])] += 1
  x = []
  y = []
  for key in lengths:
    x.append(key)
    y.append(correct[key]/lengths[key])
  return x,y

# Find the well-formed percent per length of target output
def well_formed_per_length(pred, val):
  '''
    pred.size() = [number of predictions, length of each prediction]
    val.size() = [number of targets, 1, length of each target]
    number of predictions == number of targets

    returns 
    x = list of lengths
    y = list of percent well-formed
  '''
  lengths = {}
  correct = {}
  for i in range(len(pred)):
    wrong = 0
    if len(val[i][0]) in lengths:
      lengths[len(val[i][0])] += 1
    else:
      lengths[len(val[i][0])] = 1
      correct[len(val[i][0])] = 0
    for j in range(min(len(pred[i]), len(val[i][0]))):
      if pred[i][j] != val[i][0][j]:
        wrong = 1
        break
    if wrong == 0:
      correct[len(val[i][0])] += 1
  x = []
  y = []
  for key in lengths:
    x.append(key)
    y.append(correct[key]/lengths[key])
  return x,y

# Find the exact match accuracy per length of target input
def acc_per_input(pred, val, val_set):
  '''
    pred.size() = [number of predictions, length of each prediction]
    val.size() = [number of targets, 1, length of each target]
    number of predictions == number of targets
    val_set = custom data structure of target sentences

    returns 
    x = list of lengths
    y = list of accuracies
  '''
  lengths = {}
  correct = {}
  for i in range(len(pred)):
    wrong = 0
    if len(val_set.src_sentences[i]) in lengths:
      lengths[len(val_set.src_sentences[i])] += 1
    else:
      lengths[len(val_set.src_sentences[i])] = 1
      correct[len(val_set.src_sentences[i])] = 0
    for j in range(min(len(pred[i]), len(val[i][0]))):
      if pred[i][j] != val[i][0][j]:
        wrong = 1
        break
    if wrong == 0:
      correct[len(val_set.src_sentences[i])] += 1
  x = []
  y = []
  for key in lengths:
    x.append(key)
    y.append(correct[key]/lengths[key])
  return x,y

# Find the exact match accuracy per dialog turn of target input
def acc_per_diag_turn(pred, val, val_set):
  '''
    pred.size() = [number of predictions, length of each prediction]
    val.size() = [number of targets, 1, length of each target]
    number of predictions == number of targets
    val_set = custom data structure of target sentences

    returns 
    x = list of number of dialog turns
    y = list of accuraries
  '''
  lengths = {}
  correct = {}
  for i in range(len(pred)):
    l = 0
    for k in range(len(val_set.src_sentences[i])):
      if val_set.src_sentences[i][k] == "__StartOfProgram":
        l += 1
    wrong = 0
    if l in lengths:
      lengths[l] += 1
    else:
      lengths[l] = 1
      correct[l] = 0
    for j in range(min(len(pred[i]), len(val[i][0]))):
      if pred[i][j] != val[i][0][j]:
        wrong = 1
        break
    if wrong == 0:
      correct[l] += 1
  x = []
  y = []
  for key in lengths:
    x.append(key)
    y.append(correct[key]/lengths[key])
  return x,y
