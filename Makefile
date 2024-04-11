# source: https://github.com/TheNetAdmin/Makefile-Templates/blob/master/SmallProject/Template/Makefile
# tool macros
CC :=icc 
CXX := icpc #icpc
CFLAGS := -pedantic -Wall -O3 -qopenmp
CXXFLAGS := -pedantic -Wall -O3 -qopenmp -std=c++17 -msse4.2 -axAVX,CORE-AVX2 # -Wall -O3 -fopenmp -std=c++17
DBGFLAGS := -g
COBJFLAGS := $(CFLAGS) -c

# path macros
BIN_PATH := bin
OBJ_PATH := obj
SRC_PATH := cgp

# compile macros
TARGET_NAME := cgp
TARGET := $(BIN_PATH)/$(TARGET_NAME)
TARGET_DEBUG := $(DBG_PATH)/$(TARGET_NAME)

# src files & obj files
SRC := $(SRC_PATH)/StringTemplate.cpp $(SRC_PATH)/Configuration.cpp $(SRC_PATH)/Dataset.cpp $(SRC_PATH)/Stream.cpp $(SRC_PATH)/Chromosome.cpp $(SRC_PATH)/Cgp.cpp $(SRC_PATH)/Main.cpp
OBJ := $(OBJ_PATH)/StringTemplate.o $(OBJ_PATH)/Configuration.o $(OBJ_PATH)/Dataset.o $(OBJ_PATH)/Stream.o $(OBJ_PATH)/Chromosome.o $(OBJ_PATH)/Cgp.o $(OBJ_PATH)/Main.o

# clean files list
DISTCLEAN_LIST := $(OBJ) 
CLEAN_LIST := $(TARGET) \
			  $(DISTCLEAN_LIST)

# default rule
default: makedir all

# non-phony targets
$(TARGET): $(OBJ)
	$(CXX) -o $@ $(OBJ) $(CXXFLAGS)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c*
	$(CXX) $(CXXFLAGS) $(COBJFLAGS) -o $@ $<

# phony rules
.PHONY: makedir
makedir:
	@mkdir -p $(BIN_PATH) $(OBJ_PATH) $(DBG_PATH)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)

.PHONY: distclean
distclean:
	@echo CLEAN $(DISTCLEAN_LIST)
	@rm -f $(DISTCLEAN_LIST)
