# Environment variables passed via elixir_make
# IREE_GIT_REV
# NX_IREE_SOURCE_DIR
# IREE_BUILD_TARGET
# MIX_APP_PATH

# System vars
TEMP ?= $(HOME)/.cache
IREE_REPO ?= https://github.com/iree-org/iree

IREE_NS = iree-$(IREE_GIT_REV)
NX_IREE_SOURCE_DIR ?= $(TEMP)/nx_iree/$(IREE_NS)

PRIV_DIR = $(MIX_APP_PATH)/priv

# default rule for elixir_make
ifeq ($(NX_IREE_PREFER_PRECOMPILED), true)
all: nx_iree
else
all: clone_iree install_runtime nx_iree
endif

.PHONY: clone_iree
clone_iree: $(NX_IREE_SOURCE_DIR) $(NX_IREE_SOURCE_DIR)/.nx_iree_patches_applied
	@echo "Cloned IREE into $(NX_IREE_SOURCE_DIR)"

$(NX_IREE_SOURCE_DIR):
	./scripts/clone_iree.sh $(IREE_GIT_REV) $(NX_IREE_SOURCE_DIR)

IREE_PATCH_FILES = $(wildcard patches/iree/*.patch)
IREE_PATCH_STAMP = $(NX_IREE_SOURCE_DIR)/.nx_iree_patches_applied

$(IREE_PATCH_STAMP): $(IREE_PATCH_FILES) scripts/apply_iree_patches.sh | $(NX_IREE_SOURCE_DIR)
	bash scripts/apply_iree_patches.sh "$(NX_IREE_SOURCE_DIR)" "$(abspath patches/iree)"
	touch "$@"

# Nerves sets CROSSCOMPILE (the toolchain prefix) when building for a device.
# Cross builds get their own build, install and cache directories so they
# coexist with the host build.
ifneq ($(strip $(CROSSCOMPILE)),)
IREE_BUILD_TARGET = cross
CROSS_NAME := $(notdir $(CROSSCOMPILE))
IREE_CMAKE_BUILD_DIR ?= $(abspath iree-runtime/$(CROSS_NAME)/iree-build)
IREE_RUNTIME_BUILD_DIR ?= $(abspath iree-runtime/$(CROSS_NAME)/build)
IREE_INSTALL_DIR ?= $(abspath iree-runtime/$(CROSS_NAME)/install)
NX_IREE_CACHE_DIR = cache/$(CROSS_NAME)
else
NX_IREE_CACHE_DIR = cache
endif

IREE_CMAKE_BUILD_DIR ?= $(abspath iree-runtime/iree-build)
IREE_RUNTIME_INCLUDE_PATH := $(abspath $(NX_IREE_SOURCE_DIR)/runtime/src/iree)
IREE_RUNTIME_BUILD_DIR ?= $(abspath iree-runtime/build)
IREE_INSTALL_DIR ?= $(abspath iree-runtime/host/install)

# Code generators IREE's build runs on the build machine
IREE_HOST_TOOLS_DIR = $(abspath iree-runtime/host-tools)
IREE_HOST_TOOLS_BIN = $(IREE_HOST_TOOLS_DIR)/tools
HOST_ENV = env -u CC -u CXX -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
	-u AR -u AS -u LD -u STRIP -u PKG_CONFIG_SYSROOT_DIR -u PKG_CONFIG_LIBDIR

IREE_BUILD_TARGET ?= host

BUILD_TARGET_FLAGS = -S $(abspath cmake)

CUDA_PRESENT := $(shell command -v nvcc >/dev/null 2>&1 && echo true || echo false)

ifeq ($(CUDA_PRESENT), true)
	ifeq ($(IREE_BUILD_TARGET), host)
		NIF_CFLAGS += -DCUDA_ENABLED
		CMAKE_CXX_FLAGS += -DCUDA_ENABLED
	endif
endif

BUILD_HOST_COMPILER ?= OFF
BUILD_HOST_COMPILER_FLAGS := ""

ifeq ($(BUILD_HOST_COMPILER), ON)
	BUILD_HOST_COMPILER_FLAGS += \
		-DIREE_BUILD_COMPILER=ON \
		-DIREE_INPUT_TORCH=OFF \
		-DIREE_INPUT_TOSA=OFF \
		-DIREE_BUILD_SAMPLES=OFF \
		-DIREE_BUILD_TESTS=OFF \
		-DIREE_HAL_DRIVER_DEFAULTS=OFF \
		-DIREE_BUILD_PYTHON_BINDINGS=OFF \
		-DIREE_BUILD_BINDINGS_TFLITE=OFF
endif

# apple target flags specified for xcode 15.4
ifeq ($(IREE_BUILD_TARGET), host)
else ifeq ($(IREE_BUILD_TARGET), cross)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_PROCESSOR=$(firstword $(subst -, ,$(CROSS_NAME))) \
		-DCMAKE_C_COMPILER=$(CC) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_SYSROOT=$(NERVES_SDK_SYSROOT) \
		-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
		-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
		-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
		-DIREE_HOST_BIN_DIR=$(IREE_HOST_TOOLS_BIN) \
		-DIREE_ENABLE_WERROR_FLAG=OFF
# Buildroot toolchains add a -mcpu for the device, which GCC warns conflicts
# with the -march of IREE's per-feature ukernels (selected at run time)
else ifeq ($(IREE_BUILD_TARGET), webassembly)
  BUILD_TARGET_FLAGS += \
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else ifeq ($(IREE_BUILD_TARGET), ios)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=iOS\
		-DCMAKE_OSX_DEPLOYMENT_TARGET=17.5\
		-DCMAKE_OSX_ARCHITECTURES=arm64\
		-DCMAKE_SYSTEM_PROCESSOR=arm64\
		-DCMAKE_IOS_INSTALL_COMBINED=YES\
		-DCMAKE_OSX_SYSROOT=$(shell xcodebuild -version -sdk iphoneos Path)\
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else ifeq ($(IREE_BUILD_TARGET), ios_simulator)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=iOS\
		-DCMAKE_OSX_DEPLOYMENT_TARGET=17.5\
		-DCMAKE_OSX_ARCHITECTURES=arm64\
		-DCMAKE_SYSTEM_PROCESSOR=arm64\
		-DCMAKE_IOS_INSTALL_COMBINED=YES\
		-DCMAKE_OSX_SYSROOT=$(shell xcodebuild -version -sdk iphonesimulator Path)\
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else ifeq ($(IREE_BUILD_TARGET), visionos)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=visionOS\
		-DCMAKE_OSX_DEPLOYMENT_TARGET=1.2\
		-DCMAKE_OSX_ARCHITECTURES=arm64\
		-DCMAKE_SYSTEM_PROCESSOR=arm64\
		-DCMAKE_OSX_SYSROOT=$(shell xcodebuild -version -sdk xros Path)\
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else ifeq ($(IREE_BUILD_TARGET), visionos_simulator)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=visionOS\
		-DCMAKE_OSX_DEPLOYMENT_TARGET=1.2\
		-DCMAKE_OSX_ARCHITECTURES=arm64\
		-DCMAKE_SYSTEM_PROCESSOR=arm64\
		-DCMAKE_OSX_SYSROOT=$(shell xcodebuild -version -sdk xrsimulator Path)\
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else ifeq ($(IREE_BUILD_TARGET), tvos)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=tvOS\
		-DCMAKE_OSX_DEPLOYMENT_TARGET=17.5\
		-DCMAKE_OSX_ARCHITECTURES=arm64\
		-DCMAKE_SYSTEM_PROCESSOR=arm64\
		-DCMAKE_OSX_SYSROOT=$(shell xcodebuild -version -sdk appletvos Path)\
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else ifeq ($(IREE_BUILD_TARGET), tvos_simulator)
	BUILD_TARGET_FLAGS += \
		-DCMAKE_SYSTEM_NAME=tvOS\
		-DCMAKE_OSX_DEPLOYMENT_TARGET=17.5\
		-DCMAKE_OSX_ARCHITECTURES=arm64\
		-DCMAKE_SYSTEM_PROCESSOR=arm64\
		-DCMAKE_OSX_SYSROOT=$(shell xcodebuild -version -sdk appletvsimulator Path)\
		-DIREE_HOST_BIN_DIR=$(abspath $(IREE_HOST_BIN_DIR))
else
	$(error "Unknown IREE_BUILD_TARGET: $(IREE_BUILD_TARGET), must be one of host, ios, ios_simulator, visionos, visionos_simulator, tvos, tvos_simulator")
endif

# A file target, so the runtime is rebuilt whenever its sources change
IREE_INSTALL_LIB = $(IREE_INSTALL_DIR)/libnx_iree_runtime.so

.PHONY: install_runtime
ifneq ($(strip $(IREE_HOST_BUILD_DIR)),)
install_runtime: $(IREE_HOST_INSTALL_DIR)/bin/iree-flatcc-cli $(IREE_INSTALL_LIB)
else
install_runtime: $(IREE_INSTALL_LIB)
endif

CMAKE_SOURCES = $(wildcard cmake/src/*.cc cmake/src/*.h)

$(IREE_HOST_TOOLS_BIN)/iree-c-embed-data: $(NX_IREE_SOURCE_DIR)
	$(HOST_ENV) cmake -G Ninja -B $(IREE_HOST_TOOLS_DIR) -S $(NX_IREE_SOURCE_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DIREE_BUILD_COMPILER=OFF \
		-DIREE_BUILD_TESTS=OFF \
		-DIREE_BUILD_SAMPLES=OFF \
		-DIREE_HAL_DRIVER_DEFAULTS=OFF
	$(HOST_ENV) cmake --build $(IREE_HOST_TOOLS_DIR) --target iree-flatcc-cli iree-c-embed-data

$(IREE_INSTALL_LIB): $(NX_IREE_SOURCE_DIR) $(IREE_PATCH_STAMP) $(CMAKE_SOURCES) \
		$(if $(CROSS_NAME),$(IREE_HOST_TOOLS_BIN)/iree-c-embed-data)
	$(EMCMAKE) cmake -G Ninja -B $(IREE_CMAKE_BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(IREE_CMAKE_CONFIG)\
		-DIREE_BUILD_COMPILER=OFF\
		-DIREE_RUNTIME_BUILD_DIR=$(IREE_RUNTIME_BUILD_DIR)\
		-DIREE_RUNTIME_INCLUDE_PATH=$(IREE_RUNTIME_INCLUDE_PATH)\
		-DNX_IREE_SOURCE_DIR=$(NX_IREE_SOURCE_DIR) \
		-DCMAKE_CXX_FLAGS=$(CMAKE_CXX_FLAGS) \
		$(BUILD_TARGET_FLAGS)
	cmake --build $(IREE_CMAKE_BUILD_DIR) --config $(IREE_CMAKE_CONFIG)
	cmake --install $(IREE_CMAKE_BUILD_DIR) --config $(IREE_CMAKE_CONFIG) --prefix $(IREE_INSTALL_DIR)
	touch $@

.PHONY: iree_host
ifneq ($(strip $(IREE_HOST_BUILD_DIR)),)
iree_host: $(IREE_HOST_BUILD_DIR)/bin/iree-flatcc-cli
else
iree_host:
	@echo "IREE_HOST_BUILD_DIR not set. Skipping host binaries build."
endif

$(IREE_HOST_INSTALL_DIR)/bin/iree-flatcc-cli: $(NX_IREE_SOURCE_DIR) $(CMAKE_SOURCES)
	@echo "Building IREE runtime host binaries at `$(IREE_HOST_BUILD_DIR)`."
	cmake -G Ninja -B $(IREE_HOST_BUILD_DIR) \
		-DCMAKE_INSTALL_PREFIX=$(IREE_HOST_INSTALL_DIR) \
		-DIREE_BUILD_COMPILER=$(BUILD_HOST_COMPILER) $(BUILD_HOST_COMPILER_FLAGS) \
		-DCMAKE_BUILD_TYPE=$(IREE_CMAKE_CONFIG) \
		-DCMAKE_CXX_FLAGS=$(CMAKE_CXX_FLAGS) \
		-S $(NX_IREE_SOURCE_DIR)
	cmake --build $(IREE_HOST_BUILD_DIR) --target install

### NxIREE Runtime NIF library

NX_IREE_SO ?= $(MIX_APP_PATH)/priv/libnx_iree.so
NX_IREE_CACHE_SO ?= $(NX_IREE_CACHE_DIR)/libnx_iree.so
NX_IREE_SO_LINK_PATH = $(CWD_RELATIVE_TO_PRIV_PATH)/$(NX_IREE_CACHE_SO)

NX_IREE_RUNTIME_LIB = $(NX_IREE_CACHE_DIR)/iree-runtime
NX_IREE_RUNTIME_CACHE_LIB = $(NX_IREE_RUNTIME_LIB)/libnx_iree_runtime.so
NX_IREE__IREE_RUNTIME_INCLUDE_PATH = $(NX_IREE_RUNTIME_LIB)/include
NX_IREE_RUNTIME_SO ?= $(MIX_APP_PATH)/priv/libnx_iree_runtime.so

NIF_CFLAGS += -fPIC -I$(ERTS_INCLUDE_DIR) -I$(NX_IREE__IREE_RUNTIME_INCLUDE_PATH) -Wall -std=c++17 -w

IREE_CMAKE_CONFIG ?= Release

ifdef DEBUG
	IREE_CMAKE_CONFIG = RelWithDebInfo
	NIF_CFLAGS += -g
else
	NIF_CFLAGS += -O3
endif

NIF_LDFLAGS = -L$(NX_IREE_RUNTIME_LIB) -lnx_iree_runtime -shared

# Nerves passes the sysroot in CFLAGS/LDFLAGS, which the NIF build does not use
ifneq ($(strip $(NERVES_SDK_SYSROOT)),)
	NIF_CFLAGS += --sysroot=$(NERVES_SDK_SYSROOT)
	NIF_LDFLAGS += --sysroot=$(NERVES_SDK_SYSROOT)
endif

ifeq ($(shell uname -s), Darwin)
	NIF_LDFLAGS += -flat_namespace -undefined dynamic_lookup -rpath @loader_path/iree-runtime
else
	# Use a relative RPATH, so at runtime libexla.so looks for libxla_extension.so
	# in ./lib regardless of the absolute location. This way priv can be safely
	# packed into an Elixir release. Also, we use $$ to escape Makefile variable
	# and single quotes to escape shell variable
	NIF_LDFLAGS += -Wl,-rpath,'$$ORIGIN/iree-runtime'
endif

NX_IREE_LIB_DIR = $(MIX_APP_PATH)/priv/iree-runtime
NX_IREE_LIB_LINK_PATH = $(abspath $(NX_IREE_RUNTIME_LIB))
NX_IREE_CACHE_SO_LINK_PATH = $(NX_IREE_CACHE_SO)

SOURCES = $(wildcard c_src/*.cc)
HEADERS = $(wildcard c_src/*.h)
OBJECTS = $(patsubst c_src/%.cc,$(NX_IREE_CACHE_DIR)/objs/%.o,$(SOURCES))

ifeq ($(NX_IREE_PREFER_PRECOMPILED), true)
# If we are using precompiled libnx_iree.so, we need to make sure that
# we're not trying to compile it again, which may happen due to the file
# having been recently downloaded.
# By using different nx_iree and NX_IREE_CACHE_SO rules we can ensure
# that only the precompiled .so is used directly.
nx_iree: $(NX_IREE_SO)

.PHONY: $(NX_IREE_CACHE_SO)
$(NX_IREE_CACHE_SO):
ifdef DEBUG
	@echo "Using precompiled libnx_iree.so"
endif

else

nx_iree: $(NX_IREE_RUNTIME_CACHE_LIB) $(NX_IREE_SO)

$(NX_IREE_CACHE_SO): $(OBJECTS)
	$(CXX) -shared -o $@ $^ $(NIF_LDFLAGS)

# Refresh the runtime the NIF links and loads whenever it is rebuilt
$(NX_IREE_RUNTIME_CACHE_LIB): $(IREE_INSTALL_LIB)
	rm -rf $(NX_IREE_RUNTIME_LIB)
	mkdir -p $(dir $(NX_IREE_RUNTIME_LIB))
	cp -r $(IREE_INSTALL_DIR) $(NX_IREE_RUNTIME_LIB)
endif

# Always refreshed: priv may hold entries from a build for another target
# (for a path dependency it is copied from, or links to, the source priv)
$(NX_IREE_SO): $(NX_IREE_CACHE_SO) FORCE
	@ if [ -L $(PRIV_DIR) ]; then \
		src=$$(readlink -f $(PRIV_DIR)) ; \
		rm $(PRIV_DIR) && mkdir -p $(PRIV_DIR) && cp $$src/VERSION $(PRIV_DIR)/ ; \
	fi
	@mkdir -p $(PRIV_DIR)
	@rm -rf $(NX_IREE_LIB_DIR) $(NX_IREE_SO)
	@ if [ -n "$(CROSS_NAME)" ]; then rm -f $(PRIV_DIR)/iree-compile ; fi
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ] || [ -n "$(CROSS_NAME)" ]; then \
		mkdir -p $(NX_IREE_LIB_DIR) ; \
		cp -a $(abspath $(NX_IREE_RUNTIME_CACHE_LIB)) $(NX_IREE_LIB_DIR)/ ; \
		cp -a $(abspath $(NX_IREE_CACHE_SO)) $(NX_IREE_SO) ; \
	else \
		ln -sf $(NX_IREE_LIB_LINK_PATH) $(NX_IREE_LIB_DIR) ; \
		ln -sf $(NX_IREE_CACHE_SO_LINK_PATH) $(NX_IREE_SO) ; \
	fi

.PHONY: FORCE
FORCE:


$(NX_IREE_CACHE_DIR)/objs/%.o: c_src/%.cc $(CMAKE_SOURCES) $(NX_IREE_RUNTIME_CACHE_LIB)
	@ mkdir -p $(dir $@)
	$(CXX) $(NIF_CFLAGS) -o $@ -c $<

# Print IREE Dir
.PHONY: PTD
PTD:
	@ echo $(NX_IREE_SOURCE_DIR)

.PHONY: wasm_build
webassembly:
	./scripts/build_and_package.sh --target=host --build-compiler
	./scripts/build_and_package.sh --target=webassembly

clean:
	rm -rf $(NX_IREE_CACHE_DIR)/objs
	rm -rf $(TARGET_DIR)
