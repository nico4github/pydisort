# Reproducible local development workflow for the C/CUDA extension.
#
# The Python binding includes C-DISORT headers inline, so a core-header change
# requires both the CMake libraries and the extension to be rebuilt.  Always
# use `make rebuild` rather than copying an individual shared library.

VENV := $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
VENV_BIN := $(VENV)/bin
BUILD_DIR := $(CURDIR)/build
PACKAGE_DIR := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/pydisort
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
CUDA_ARCH ?= 90

.PHONY: configure build extension install rebuild verify

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
		-DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)

build:
	cmake --build $(BUILD_DIR) --parallel

extension:
	@if test ! -f python/pydisort$(EXT_SUFFIX) || \
		find python/csrc python/disort src disort cdisort213 -type f -newer python/pydisort$(EXT_SUFFIX) -print -quit | grep -q .; then \
		echo 'Rebuilding pydisort extension because a native source changed'; \
		PATH="$(VENV_BIN):$$PATH" WORKSPACE="$(CURDIR)" $(PYTHON) setup.py build_ext --inplace; \
	else \
		echo 'pydisort extension is current'; \
	fi

install:
	test -f python/pydisort$(EXT_SUFFIX)
	test -d $(PACKAGE_DIR)/lib
	cp python/pydisort$(EXT_SUFFIX) $(PACKAGE_DIR)/pydisort$(EXT_SUFFIX)
	cp $(BUILD_DIR)/lib/libdisort_release.so $(PACKAGE_DIR)/lib/libdisort_release.so
	cp $(BUILD_DIR)/lib/libdisort_cuda_release.so $(PACKAGE_DIR)/lib/libdisort_cuda_release.so
	$(MAKE) verify

rebuild: build extension install

verify:
	$(PYTHON) -c 'import pydisort; print(pydisort.__file__)'
