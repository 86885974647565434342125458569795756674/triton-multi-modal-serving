#!/bin/bash

cp dynamic_batch_scheduler.cc dynamic_batch_scheduler.h tritonserver.cc tritonserver_stub.cc server.h server.cc model.h scheduler.h /tmp/tritonbuild/tritonserver/build/_deps/repo-core-src/src/

cp tritonserver.cc /tmp/tritonbuild/tritonserver/build/triton-server/_deps/repo-core-src/src/

cp tritonserver_stub.cc /tmp/tritonbuild/tritonserver/build/triton-server/_deps/repo-core-src/src/

cp tritonserver.h /tmp/tritonbuild/tritonserver/build/_deps/repo-core-src/include/triton/core/

cp tritonserver.h /tmp/tritonbuild/tritonserver/build/triton-server/_deps/repo-core-src/include/triton/core/

cp tritonserver_pybind.cc /tmp/tritonbuild/tritonserver/build/triton-server/_deps/repo-core-src/python/tritonserver/_c/

cp tritonserver_pybind.cc /tmp/tritonbuild/tritonserver/build/_deps/repo-core-src/python/tritonserver/_c/
