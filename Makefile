CFG = config/config-defaults.yaml
CODE = None
LOGDIR = ./logs
CKPT = None
EXTRA = None
INIT = False

all:
	@echo please use \"make train\" or other ...

train:
	python ${CODE} --config_path ${CFG} --init_train_flag ${INIT} --pre_train_weight ${CKPT}

test:
	python ${CODE} --config ${CFG} --stage test --ckpt ${CKPT}

infer:
	python ${CODE} --config None --stage infer --ckpt ${CKPT} --extra ${EXTRA}

export:
	python ${CODE} --config None --stage export --ckpt ${CKPT} --extra ${EXTRA}

tensorboard:
	tensorboard --logdir ${LOGDIR} --bind_all