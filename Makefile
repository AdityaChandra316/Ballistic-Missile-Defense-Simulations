PYTHON      := python3
CXX         := clang++
CXXFLAGS    := -O3 -std=c++17 -shared -fPIC

MODULE      := trajectory
SRC         := trajectory.cpp
EXT_SUFFIX  := $(shell $(PYTHON)-config --extension-suffix)
TARGET      := $(MODULE)$(EXT_SUFFIX)

PYBIND_INC  := $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS     := $(shell $(PYTHON)-config --embed --ldflags)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND_INC) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
