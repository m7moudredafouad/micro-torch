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
cd $output_dir
make
error=$?
if [ $error == 0 ]; then
    echo ">>> Done."
else
    echo ">>> Make error $error"
fi
cd -

if [[ "$OSTYPE" == "msys" ]]; then
    ln -s $(pwd)/$output_dir/libs/glog/libglogd.dll $(pwd)/$output_dir
fi