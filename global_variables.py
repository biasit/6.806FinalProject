# Length stuff
MAX_SRC_LENGTH = 200
MAX_TRG_LENGTH = 100

# Device
#device = 'cuda'
device = "cpu" # for debugging

# Tokens (no retrieval)
START_TOKEN = "__START_PREDICT"
END_TOKEN = "__EndOfProgram"
PAD_INDEX = 0
PAD_TOKEN = "PAD"
UNK_INDEX = 1
UNK_TOKEN = "<unk>"
START_INDEX = 2
END_INDEX = 3

SAVED_INDEXES = {PAD_TOKEN: PAD_INDEX, UNK_TOKEN: UNK_INDEX,
                 START_TOKEN: START_INDEX, END_TOKEN: END_INDEX}

# Tokens (retrieval)
RETR_TOKEN = "RETRIEVED"
RETR_INDEX = 4
MAX_SRC_LENGTH_RETR = MAX_SRC_LENGTH + MAX_TRG_LENGTH
SAVED_INDEXES_RETR = {PAD_TOKEN: PAD_INDEX,
                 UNK_TOKEN: UNK_INDEX,
                 START_TOKEN: START_INDEX,
                 END_TOKEN: END_INDEX,
                 RETR_TOKEN: RETR_INDEX}
# Train Hyperparameters
GLOVE_DIM = 300
EMBED_SIZE = GLOVE_DIM
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
BATCH_SIZE = 64
MODEL_FOLDER = './train_models/'

# Plotting
BASE_COLOR = 320 # high activation seems to be ~ 0.8
