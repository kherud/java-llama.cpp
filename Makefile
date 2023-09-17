include Makefile.common

# g++     -I. -O3 -m64 -I"/usr/lib/jvm/zulu-17/include" -I"/usr/lib/jvm/zulu-17/include/linux"     -shared -fPIC -fvisibility=hidden -static-libgcc -pthread -lstdc++ -lllama     -L.     -o libjllama.so     jllama.cpp common.cpp && mv -f libjllama.so ../../../../src/main/resources/de/kherud/llama/Linux/x86_64/

# set up some variables
DOCKER_RUN_OPTS=--rm
CODESIGN:=docker run $(DOCKER_RUN_OPTS) -v $$PWD:/workdir gotson/rcodesign sign

NATIVE_DIR := src/main/resources/de/kherud/llama/$(OS_NAME)/$(OS_ARCH)

LLAMA_SRC := $(TARGET)/llama
LLAMA_HEADER := $(LLAMA_SRC)/llama.h
JLLAMA_HEADER := $(LLAMA_SRC)/jllama.h
LLAMA_INCLUDE := $(shell dirname "$(LLAMA_HEADER)")
LLAMA_LIB := $(LLAMA_SRC)/build/$(LLAMA_LIB_NAME)
JLLAMA_LIB := $(LLAMA_SRC)/build/$(JLLAMA_LIB_NAME)
LLAMA_CMAKE_ARGS :=

CCFLAGS := -I$(LLAMA_INCLUDE) $(CCFLAGS)

# download llama.cpp sources or pull new changes
$(LLAMA_SRC):
	@mkdir -p $@
	@if [ -d "$@/.git" ]; then \
		cd $@ && git pull; \
	else \
		git clone https://github.com/ggerganov/llama.cpp.git $@; \
	fi

# build the llama.cpp shared library
$(LLAMA_LIB): $(LLAMA_SRC)
	(cd $(LLAMA_SRC) && \
		mkdir -p $(LLAMA_SRC)\build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=ON $(LLAMA_CMAKE_ARGS) && \
		cmake --build . --config Release)

# create the jni header
$(JLLAMA_HEADER): src/main/java/de/kherud/llama/LlamaModel.java
	mvn compile
	mv $(LLAMA_SRC)/de_kherud_llama_LlamaModel.h $@

# build the jni shared library
$(JLLAMA_LIB): $(JLLAMA_HEADER)
	$(CC) \
		$(CCFLAGS) \
		$(LINKFLAGS) \
		-L $(LLAMA_SRC)/build \
		-I $(LLAMA_SRC) \
		-I $(LLAMA_SRC)/common \
		-I $(LLAMA_SRC)/examples/server \
		-o $@ \
		src/main/cpp/jllama.cpp \
		$(LLAMA_SRC)/common/common.cpp \
		$(LLAMA_SRC)/common/grammar-parser.cpp

build: $(LLAMA_LIB) $(JLLAMA_LIB)
	mkdir -p $(NATIVE_DIR)
	mv $(LLAMA_LIB) $(NATIVE_DIR)
	mv $(JLLAMA_LIB) $(NATIVE_DIR)

build-jni: $(JLLAMA_LIB)
	mv $(JLLAMA_LIB) $(NATIVE_DIR)

build-jni-header: $(JLLAMA_HEADER)

clean-build:
	@rm -rf $(LLAMA_SRC)/build
	@rm -rf $(TARGET)/classes
	@rm -rf $(TARGET)/test-classes

clean-llama:
	@rm -rf $(LLAMA_SRC)
