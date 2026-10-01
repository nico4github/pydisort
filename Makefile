# Reproducible local development workflow for the C/CUDA extension.
#
# The Python binding includes C-DISORT headers inline, so a core-header change
# requires both the CMake libraries and the extension to be rebuilt.  Always
# use plain `make` rather than copying an individual shared library.

VENV := $(CURDIR)/.venv
PYTHON := $(VENV)/bin/python
VENV_BIN := $(VENV)/bin
BUILD_DIR := $(CURDIR)/build
PACKAGE_DIR := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/pydisort
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
CUDA_ARCH ?= 90

.PHONY: configure build clean-extension extension install rebuild verify

.DEFAULT_GOAL := rebuild

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
		-DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)

build: configure
	cmake --build $(BUILD_DIR) --parallel

clean-extension:
	$(PYTHON) setup.py clean --all
	rm -rf $(BUILD_DIR)/temp.* $(BUILD_DIR)/lib.*

extension: clean-extension
	PATH="$(VENV_BIN):$$PATH" WORKSPACE="$(CURDIR)" $(PYTHON) setup.py build_ext --inplace --force
	test -f python/pydisort$(EXT_SUFFIX)

install:
	test -f python/pydisort$(EXT_SUFFIX)
	test -d $(PACKAGE_DIR)/lib
	cp python/pydisort$(EXT_SUFFIX) $(PACKAGE_DIR)/pydisort$(EXT_SUFFIX)
	cp python/*.py python/*.pyi $(PACKAGE_DIR)/
	cp $(BUILD_DIR)/lib/libdisort_release.so $(PACKAGE_DIR)/lib/libdisort_release.so
	cp $(BUILD_DIR)/lib/libdisort_cuda_release.so $(PACKAGE_DIR)/lib/libdisort_cuda_release.so
	$(MAKE) verify

rebuild: build extension
	$(MAKE) install

verify:
	$(PYTHON) -c 'import pydisort; assert "site-packages/pydisort" in pydisort.__file__; print(pydisort.__file__)'
