LIBS_RYSY_PATH=$(HOME)/libs/rysy

export LIBS_RYSY_PATH

all:
	cd libs_dqn && make -j4
	cd libs_dqn_python && make -j4
	cd dqn_test && make -j4

clean:
	cd libs_dqn && make clean
	cd libs_dqn_python && make clean
	cd dqn_test && make clean
