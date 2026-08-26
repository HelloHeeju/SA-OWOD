def add_config(cfg):
    cfg.MODEL.NUM_CLASSES = 81
    cfg.MODEL.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.NHEADS = 8
    cfg.MODEL.DROPOUT = 0.0
    cfg.MODEL.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ACTIVATION = 'relu'
    cfg.MODEL.HIDDEN_DIM = 256
    cfg.MODEL.NUM_CLS = 1
    cfg.MODEL.NUM_REG = 3
    cfg.MODEL.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.NUM_DYNAMIC = 2
    cfg.MODEL.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.CLASS_WEIGHT = 2.0
    cfg.MODEL.NC = True
    cfg.MODEL.NC_WEIGHT = 0.1
    cfg.MODEL.GIOU_WEIGHT = 2.0
    cfg.MODEL.L1_WEIGHT = 5.0
    cfg.MODEL.DEEP_SUPERVISION = True
    cfg.MODEL.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.ALPHA = 0.25
    cfg.MODEL.GAMMA = 2.0
    cfg.MODEL.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.OTA_K = 5
    cfg.MODEL.FORWARD_K = 10

    # WARM_UP
    cfg.MODEL.CHANGE_START = 0

    # Diffusion
    cfg.MODEL.SNR_SCALE = 2.0
    cfg.MODEL.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.USE_NMS = True
    cfg.MODEL.M_STEP = 20
    cfg.MODEL.SAMPLING_METHOD = 'Random_'

    # Disentanglement
    cfg.MODEL.DISENTANGLED = 2  # 0: RandBox, 1: separate head, 2: feature orthogonality
    cfg.MODEL.DECORR_WEIGHT = 1.  # weight for prediction decorrelation loss

    # Combinatorial CROWD loss
    cfg.MODEL.CROWD = False
    cfg.MODEL.CROWD_IOD = False # TODO: Make a unified parameter (later)
    cfg.MODEL.CROWD_WEIGHT = 0.0  # weight for CROWD loss (uses submod CG)
    cfg.MODEL.CROWD_FUNCTION = "GCCG" # submodular function used as CROWD loss
    cfg.MODEL.CROWD_DIVERSITY = 0.5
    cfg.MODEL.CROWD_PRIVATE_HARDNESS = 1.0
    cfg.MODEL.CROWD_BALANCE_WEIGHT = 1.0 

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # OW EVALUATION
    cfg.TEST.PREV_INTRODUCED_CLS = 0
    cfg.TEST.CUR_INTRODUCED_CLS = 20
    cfg.TEST.PREV_CLASSES = ()  # previously seen classes
    cfg.TEST.MASK = 1  # 0: no mask, 1: mask unseen classes, 2: mask prev and unseen classes
    cfg.TEST.SCORE_THRESH = 0.2
    cfg.TEST.UNKNOWN_THRE_TEST = 0.5

    # Mine unknowns when true at the end of the training/eval round
    cfg.DISCOVER_UNKNOWN = False
    cfg.DISCOVER_FUNCTION_NAME = "GCCG"
    cfg.DISCOVER_STORE_PATH = "" # this should be typically empty
    
    # Dataset setting
    cfg.DATASETS.SETTING = 'easy'  # 'easy', 'hard', 'extreme', 'keep1', 'coco50missp'

    # Pseudo-label for unlabeled knowns
    cfg.MODEL.SCORE_THRESHOLD = 0.5
    cfg.MODEL.GT_IOU_THRESHOLD = 0.4
    cfg.MODEL.DSTG_OBJ_THRESHOLD = 0.2
    
    # Pseudo-label for unknowns
    cfg.MODEL.D2TG_SIM_THRESHOLD = 0.95
    cfg.MODEL.D2TG_OBJ_THRESHOLD = 0.2