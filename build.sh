#!/bin/bash
output_dir=build

mkdir -p $output_dir

if [[ "$OSTYPE" == "msys" ]]; then
    cmake -S . -B $output_dir -DCMAKE_BUILD_TYPE=Debug -G "MSYS Makefiles"
else
    cmake -S . -B $output_dir -DCMAKE_BUILD_TYPE=Debug
fi

error=$?
if [ $error != 0 ]; then
    echo ">>> Cmake error $error couldn't build the project"
    exit $error
fi

echo ">>> Building..."
cmake --build $output_dir

error=$?
if [ $error != 0 ]; then
    echo ">>> Make error $error"
    exit $error
fi

echo ">>> Done."