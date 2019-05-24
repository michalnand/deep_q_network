ELF = bin/dqn_test.elf

LIBS_PATH=$(HOME)/libs

INC+= -I.
INC+= -I$(LIBS_PATH)/deep_q_network/libs_dqn
INC+= -I$(LIBS_PATH)/rysy/libs_nn
INC+= -I$(LIBS_PATH)/rysy/libs_common
INC+= -I$(LIBS_PATH)/rysy/libs_experiment


LIBS+= $(LIBS_PATH)/deep_q_network/libs_dqn/libs_dqn.a
LIBS+= $(LIBS_PATH)/rysy/libs_nn/libs_nn.a
LIBS+= $(LIBS_PATH)/rysy/libs_common/libs_common.a
LIBS+= $(LIBS_PATH)/rysy/libs_experiment/libs_experiment.a

CC = g++
CU = nvcc
LD = g++
OBJCOPY = objcopy
OBJDUMP = objdump
SIZE = size

# DEGUBFLAGS = -g -pg

CFLAGS = $(DEGUBFLAGS) -O2 -Os -s -Wall -Wextra -pedantic -std=c++11 $(INC)
CUDAFLAGS = -O3 -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC -std=c++11 $(INC)

LDFLAGS = $(DEGUBFLAGS) -lstdc++fs -lm -lpthread -pthread -std=c++11 -lglut -lGL -lGLU -lX11 -ljsoncpp -L/usr/local/cuda/lib64 -lcuda -lcudart


SRCS+= $(shell find -name '*.cpp')
SRCS_CUDA+= $(shell find -name '*.cu')


OBJS = $(SRCS:.cpp=.o)
OBJS_CUDA = $(SRCS_CUDA:.cu=.oc)

all: $(ELF)
	$(SIZE) $(ELF)

$(ELF): $(LIBS) $(OBJS) $(OBJS_CUDA)
	$(LD) -o $@ $(OBJS) $(OBJS_CUDA) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.oc: %.cu
	$(CU) -c $(CUDAFLAGS) $< -o $@

clean:
	rm $(OBJS) -f
	rm $(OBJS_CUDA) -f
	rm $(ELF) -f
