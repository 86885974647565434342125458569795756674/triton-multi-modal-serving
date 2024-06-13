#!/bin/bash

cp dynamic_batch_scheduler.cc dynamic_batch_scheduler.h tritonserver.cc tritonserver_stub.cc server.cc model.h /tmp/tritonbuild/tritonserver/build/_deps/repo-core-src/src/

cp tritonserver.h /tmp/tritonbuild/tritonserver/build/_deps/repo-core-src/include/triton/core/

cp tritonserver_pybind.cc /tmp/tritonbuild/tritonserver/build/_deps/repo-core-src/python/tritonserver/_c/
